from .common_utils_mbyolo import *
import torch.nn.functional as F

__all__ = ("VSSBlock", "SimpleStem", "VisionClueMerge", "XSSBlock")


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # 捕捉垂直特征
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # 捕捉水平特征

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 沿两个方向分别池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_w * a_h
    
class CoordRGBlock(nn.Module):
    """
    集成坐标注意力的前馈网络
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        
        # 新增：坐标注意力
        self.coord_att = CoordAtt(hidden_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 1. Expand & Split
        x, v = self.fc1(x).chunk(2, dim=1)
        
        # 2. Local Conv + Attention + Residual
        x_res = self.dwconv(x)
        x_res = self.coord_att(x_res) # 关键：让网络关注“根”的方向
        x = self.act(x_res + x) * v
        
        # 3. Project Out
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    

class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpectralRGBlock(nn.Module):
    """
    Fused Block combining RGBlock (Local Spatial) and EDFFN (Patch-based Spectral).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., channels_first=False, patch_size=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_features = int(2 * hidden_features / 3) 
        
        self.fc1 = nn.Conv2d(in_features, self.mid_features * 2, kernel_size=1)

        self.dwconv = nn.Conv2d(self.mid_features, self.mid_features, kernel_size=3, 
                                stride=1, padding=1, bias=True, groups=self.mid_features)

        self.patch_size = patch_size
        self.fft_filter = nn.Parameter(
            torch.ones((self.mid_features, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )

        self.act = act_layer()
        self.fc2 = nn.Conv2d(self.mid_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # --- [Fix 1] 记录原始数据类型 (e.g., torch.float16) ---
        original_dtype = x.dtype 

        # 1. Expand and Split
        x_path, v_gate = self.fc1(x).chunk(2, dim=1)

        # 2. Spatial Processing
        x_spatial = self.dwconv(x_path) + x_path

        # 3. Spectral Processing
        B, C, H, W = x_path.shape
        
        h_n = (self.patch_size - H % self.patch_size) % self.patch_size
        w_n = (self.patch_size - W % self.patch_size) % self.patch_size
        
        x_padded = F.pad(x_path, (0, w_n, 0, h_n), mode='reflect')
        
        x_patches = rearrange(x_padded, 'b c (h p1) (w p2) -> b c h w p1 p2', 
                              p1=self.patch_size, p2=self.patch_size)
        
        # FFT 必须在 FP32 下进行
        x_fft = torch.fft.rfft2(x_patches.float())
        
        x_fft = x_fft * self.fft_filter
        
        x_spectral_patches = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        
        x_spectral = rearrange(x_spectral_patches, 'b c h w p1 p2 -> b c (h p1) (w p2)', 
                               p1=self.patch_size, p2=self.patch_size)
        
        x_spectral = x_spectral[:, :, :H, :W]

        # --- [Fix 2] 将频域分支的结果转回原始类型 (FP32 -> FP16) ---
        if x_spectral.dtype != original_dtype:
            x_spectral = x_spectral.to(original_dtype)

        # 4. Fusion (现在 x_spatial 和 x_spectral 类型一致了)
        x_fused = x_spatial + x_spectral

        # 5. Activation & Gating
        x = self.act(x_fused) * v_gate
        
        # 6. Output Projection
        x = self.drop(x)
        x = self.fc2(x) # 这里不再会报错，因为 x 已经是 FP16 了
        x = self.drop(x)
        
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SplitSpectralRGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, channels_first=False, drop=0., patch_size=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.mid_features = int(2 * hidden_features / 3) 
        if self.mid_features % 2 != 0:
            self.mid_features += 1
            
        self.half_dim = self.mid_features // 2

        self.fc1 = nn.Conv2d(in_features, self.mid_features * 2, kernel_size=1)

        self.dwconv = nn.Conv2d(self.half_dim, self.half_dim, kernel_size=3, 
                                stride=1, padding=1, bias=True, groups=self.half_dim)

        self.patch_size = patch_size
        
        # 定义为普通的 Parameter (可能会被 AMP 转为 FP16)
        self.fft_filter_real = nn.Parameter(
            torch.ones((self.half_dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.fft_filter_imag = nn.Parameter(
            torch.zeros((self.half_dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )

        self.act = act_layer()
        self.fc2 = nn.Conv2d(self.mid_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x
        original_dtype = x.dtype 
        
        # 1. Expand & GLU Split
        x_path, v_gate = self.fc1(x).chunk(2, dim=1)

        # 2. Channel Split
        x_spatial_in, x_spectral_in = torch.split(x_path, self.half_dim, dim=1)

        # --- Branch A: Spatial ---
        x_spatial_out = self.dwconv(x_spatial_in) + x_spatial_in

        # --- Branch B: Spectral ---
        B, C, H, W = x_spectral_in.shape
        h_n = (self.patch_size - H % self.patch_size) % self.patch_size
        w_n = (self.patch_size - W % self.patch_size) % self.patch_size
        x_padded = F.pad(x_spectral_in, (0, w_n, 0, h_n), mode='reflect')
        
        x_patches = rearrange(x_padded, 'b c (h p1) (w p2) -> b c h w p1 p2', 
                              p1=self.patch_size, p2=self.patch_size)
        
        # --- [Fix 关键点] ---
        # 即使参数被转成了 fp16，合成复数进行 FFT 计算时必须强转回 float32
        fft_filter = torch.complex(self.fft_filter_real.float(), self.fft_filter_imag.float())
        
        # Input 转 float32 进行 FFT
        x_fft = torch.fft.rfft2(x_patches.float())
        
        # 复数乘法 (ComplexFloat * ComplexFloat)
        x_fft = x_fft * fft_filter
        
        # 逆变换
        x_spectral_patches = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        
        x_spectral_out = rearrange(x_spectral_patches, 'b c h w p1 p2 -> b c (h p1) (w p2)', 
                               p1=self.patch_size, p2=self.patch_size)
        x_spectral_out = x_spectral_out[:, :, :H, :W]
        
        # 恢复原始精度 (如 FP16)
        if x_spectral_out.dtype != original_dtype:
            x_spectral_out = x_spectral_out.to(original_dtype)

        # 3. Concat
        x_fused = torch.cat([x_spatial_out, x_spectral_out], dim=1)

        # 4. Act & Gate
        x = self.act(x_fused) * v_gate
        
        # 5. Output Projection
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        # 6. Residual
        if self.in_features == self.out_features:
            x = x + shortcut
            
        return x


class ChannelAttention(nn.Module):
    """
    轻量级通道注意力模块 (SE-style)
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享感知层
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    空间注意力模块 (CBAM-style)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度进行 AvgPool 和 MaxPool
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return x * self.sigmoid(x_out)

class HybridSpectralAttentionBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, channels_first=False, drop=0., patch_size=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 中间维度
        self.mid_features = int(2 * hidden_features / 3) 
        if self.mid_features % 2 != 0: self.mid_features += 1
        self.half_dim = self.mid_features // 2

        # 1. 扩展层
        self.fc1 = nn.Conv2d(in_features, self.mid_features * 2, kernel_size=1)

        # 2. 局部空间分支 (Local Branch)
        # 使用 Depthwise Conv 提取局部特征
        self.dwconv = nn.Conv2d(self.half_dim, self.half_dim, kernel_size=3, 
                                stride=1, padding=1, bias=True, groups=self.half_dim)

        # 3. 频域全局分支 (Spectral Branch)
        self.patch_size = patch_size
        # 实部和虚部参数分离，适配 AMP
        self.fft_filter_real = nn.Parameter(
            torch.ones((self.half_dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.fft_filter_imag = nn.Parameter(
            torch.zeros((self.half_dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )

        # 4. 注意力机制 (Attention)
        # 针对融合后的特征进行注意力加权
        self.ca = ChannelAttention(self.mid_features)
        self.sa = SpatialAttention()

        self.act = act_layer()
        self.fc2 = nn.Conv2d(self.mid_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x
        original_dtype = x.dtype 
        
        # A. Expand & Split
        # x_path 用于特征提取, v_gate 用于 GLU 门控
        x_path, v_gate = self.fc1(x).chunk(2, dim=1)
        
        # 将通道一分为二
        x_spatial_in, x_spectral_in = torch.split(x_path, self.half_dim, dim=1)

        # B. Branch 1: Local Spatial Processing
        x_spatial_out = self.dwconv(x_spatial_in) + x_spatial_in

        # C. Branch 2: Global Spectral Processing
        B, C, H, W = x_spectral_in.shape
        h_n = (self.patch_size - H % self.patch_size) % self.patch_size
        w_n = (self.patch_size - W % self.patch_size) % self.patch_size
        x_padded = F.pad(x_spectral_in, (0, w_n, 0, h_n), mode='reflect')
        
        x_patches = rearrange(x_padded, 'b c (h p1) (w p2) -> b c h w p1 p2', 
                              p1=self.patch_size, p2=self.patch_size)
        
        # FFT (Force Float32 for AMP safety)
        fft_filter = torch.complex(self.fft_filter_real.float(), self.fft_filter_imag.float())
        x_fft = torch.fft.rfft2(x_patches.float())
        x_fft = x_fft * fft_filter
        x_spectral_patches = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        
        x_spectral_out = rearrange(x_spectral_patches, 'b c h w p1 p2 -> b c (h p1) (w p2)', 
                               p1=self.patch_size, p2=self.patch_size)
        x_spectral_out = x_spectral_out[:, :, :H, :W]
        
        if x_spectral_out.dtype != original_dtype:
            x_spectral_out = x_spectral_out.to(original_dtype)

        # D. Feature Fusion
        x_fused = torch.cat([x_spatial_out, x_spectral_out], dim=1)

        # E. Attention Refinement (新增的关键部分)
        # 1. 通道注意力：平衡局部特征和频域特征的权重
        x_att = self.ca(x_fused)
        # 2. 空间注意力：聚焦目标位置，抑制背景
        x_att = self.sa(x_att)
        
        # F. Activation & Gating
        # 融合注意力后的特征与门控信号相乘
        x = self.act(x_att) * v_gate
        
        # G. Output Projection
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        # H. Global Residual
        if self.in_features == self.out_features:
            x = x + shortcut
            
        return x
    

class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = SplitSpectralRGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, patch_size=8)   # patch_size=8

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = SplitSpectralRGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False, patch_size=8)   # patch_size=8

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN  
        return x


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)

# import torch
# import torch.nn as nn
# import torchvision.ops
# import math

# class SPDConv(nn.Module):
#     """
#     Space-to-Depth Convolution (无损下采样)
#     原理：将空间维度(H, W)的信息切块移动到通道维度(C)，避免 stride=2 卷积带来的像素丢失。
#     对于细小的玉米根系检测，这能显著保留边缘特征。
#     """
#     def __init__(self, in_channels, out_channels, dimension=2, kernel_size=1):
#         super().__init__()
#         self.d = dimension
#         # 下采样因子为 d，则 H, W 变为 H/d, W/d，通道数变为 C * d^2
#         self.in_features = in_channels * (self.d ** 2)
        
#         # 变换后的特征融合
#         # 通常使用 1x1 或 3x3 卷积将通道数映射到目标维度
#         # 注意：这里 stride 必须为 1，因为下采样已经在 space_to_depth 步骤完成了
#         self.conv = nn.Conv2d(self.in_features, out_channels, 
#                               kernel_size=kernel_size, stride=1, 
#                               padding=kernel_size//2, bias=False)

#     def forward(self, x):
#         # x: [B, C, H, W]
#         b, c, h, w = x.shape
#         # 1. 切分为块: [B, C, H/2, 2, W/2, 2]
#         x = x.view(b, c, h // self.d, self.d, w // self.d, self.d)
#         # 2. 堆叠到通道: [B, C, 2, 2, H/2, W/2] -> [B, C*4, H/2, W/2]
#         x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, c * (self.d ** 2), h // self.d, w // self.d)
#         # 3. 卷积融合
#         return self.conv(x)


# class DeformConv(nn.Module):
#     """
#     Modulated Deformable Convolution (DCNv2)
#     能够自适应地学习偏移量，使卷积核呈现"蛇形"或"弧形"分布，
#     非常适合捕捉弯曲的根系特征。
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
#         super(DeformConv, self).__init__()
#         self.stride = stride
#         self.padding = padding
#         self.kernel_size = kernel_size
        
#         # 1. 偏移量和掩码生成器
#         # 输出通道 = 2 * k * k (x,y坐标偏移) + 1 * k * k (权重掩码)
#         self.offset_mask_conv = nn.Conv2d(
#             in_channels, 
#             3 * kernel_size * kernel_size, 
#             kernel_size=kernel_size, 
#             stride=stride, 
#             padding=padding, 
#             bias=True
#         )
        
#         # 2. 实际卷积权重
#         self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        
#         if bias:
#             self.bias = nn.Parameter(torch.empty(out_channels))
#         else:
#             self.register_parameter('bias', None)
            
#         self.reset_parameters()

#     def reset_parameters(self):
#         # 初始化卷积权重
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)
        
#         # 【关键】初始化偏移量为 0
#         # 这样在训练初期，它的行为等同于普通卷积，随着 Loss 下降逐渐学会"变形"
#         nn.init.constant_(self.offset_mask_conv.weight, 0)
#         nn.init.constant_(self.offset_mask_conv.bias, 0)

#     def forward(self, x):
#         # 计算偏移和掩码
#         out = self.offset_mask_conv(x)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
        
#         # 调用 torchvision 算子
#         return torchvision.ops.deform_conv2d(
#             x, offset, self.weight, self.bias, 
#             self.stride, self.padding, dilation=1, mask=mask
#         )


# class SimpleStem(nn.Module):
#     """
#     修改版 Stem：集成 SPD (保细节) 和 动态蛇形卷积 (抓形状)
#     """
#     def __init__(self, inp, embed_dim, ks=3):
#         super().__init__()
#         self.hidden_dims = embed_dim // 2
        
#         # 1. 初始下采样：使用 SPDConv 替代普通 stride=2 卷积
#         # 注意：SPDConv 内部已经包含了下采样逻辑，所以这里不需要 stride 参数
#         self.conv1 = nn.Sequential(
#             SPDConv(inp, self.hidden_dims, dimension=2, kernel_size=ks), 
#             nn.BatchNorm2d(self.hidden_dims),
#             nn.GELU()
#         )
        
#         # 2. 动态蛇形卷积 (模拟)
#         # 使用 DCNv2 提取弯曲特征，保持分辨率不变 (Stride=1)
#         self.snake_op = DeformConv(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1)
#         self.snake_bn = nn.BatchNorm2d(self.hidden_dims)
#         self.snake_act = nn.SiLU()

#         # 3. 二次下采样：使用普通卷积完成剩余的下采样任务
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=ks//2, bias=False),
#             nn.BatchNorm2d(embed_dim),
#             nn.SiLU()
#         )

#     def forward(self, x):
#         # Step 1: SPD Downsample (H/2, W/2) - 保留微小根系像素
#         x = self.conv1(x)
        
#         # Step 2: Extract Curved Features (Residual) - 适应根系弯曲
#         # 残差连接让特征提取更稳定
#         feat = self.snake_op(x)
#         feat = self.snake_bn(feat)
#         feat = self.snake_act(feat)
#         x = x + feat
        
#         # Step 3: Standard Downsample (H/4, W/4)
#         x = self.conv2(x)
#         return x


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)
