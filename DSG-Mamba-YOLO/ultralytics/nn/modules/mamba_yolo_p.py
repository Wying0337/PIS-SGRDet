from .common_utils_mbyolo import *

__all__ = ("VSSBlock", "SimpleStem", "VisionClueMerge", "XSSBlock")


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
#====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple

class SimpleChannelAtt(nn.Module):
    """
    SE-Block (Squeeze-and-Excitation): 最经典的简单通道注意力
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 1. Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Excitation: 两个全连接层 (用 1x1 Conv 实现)
        # 降维比例 reduction 默认 16，如果通道数很少(如<32)，建议改为 4 或 8
        mid_channels = max(4, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid() # 输出 0~1 的权重
        )

    def forward(self, x):
        # 原始特征 * 通道权重
        return x * self.fc(self.avg_pool(x))

class Mlp(nn.Module):
    """ 简单的 MLP，用于生成动态路由权重 """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class RobustSpectralBranch(nn.Module):
    def __init__(self, dim, size=14, num_filters=4, reweight_expansion_ratio=0.25):
        super().__init__()
        self.num_filters = num_filters
        self.dim = dim
        self.size = size 
        self.filter_size = size // 2 + 1 
        
        # 1. 动态路由生成器
        self.reweight = Mlp(dim, int(dim * reweight_expansion_ratio), num_filters * dim)
        
        # 2. 复数权重
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        # 记录原始精度 (FP16)
        original_dtype = x.dtype 
        B, C, H, W = x.shape
        
        # --- A. 动态路由计算 ---
        # 1. 全局平均池化
        # 保持与输入 x 相同的精度 (FP16)，以便通过 FP16 的 MLP
        x_gap = x.mean(dim=(2, 3)) 
        
        # 2. 通过 MLP
        routeing = self.reweight(x_gap)
        
        # 3. 转为 Float32 进行 Softmax
        # [状态] routeing 现在是 Float32
        routeing = routeing.to(torch.float32).view(B, self.num_filters, C).softmax(dim=1)
        
        # --- B. FFT 变换 ---
        x_fft = torch.fft.rfft2(x.float(), dim=(2, 3), norm='ortho')
        
        # --- C. 权重处理 (关键修复) ---
        
        # [Fix] 显式将权重参数转为 Float32
        # 在验证模式下，self.complex_weights 是 FP16，必须转回 FP32 才能和 routeing 匹配
        complex_weights_fp32 = self.complex_weights.to(torch.float32)
        
        # 使用转为 FP32 后的权重进行变形
        weight_params = complex_weights_fp32.permute(2, 3, 0, 1).contiguous()
        weight_params = weight_params.view(self.num_filters * 2, self.size, self.filter_size)
        
        target_h, target_w = H, W // 2 + 1
        
        # 插值
        if weight_params.shape[-2:] != (target_h, target_w):
            weight_params = F.interpolate(
                weight_params.unsqueeze(0), 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=True
            ).squeeze(0)
            
        weight_params = weight_params.view(self.num_filters, 2, target_h, target_w)
        
        # --- D. 安全的实数 Einsum ---
        # 拆解为实部和虚部 (此时它们都是 Float32)
        weight_real = weight_params[:, 0, :, :] 
        weight_imag = weight_params[:, 1, :, :] 
        
        # Einsum: Float32 * Float32 -> Float32 (不会再报错)
        f_real = torch.einsum('bfc,fhw->bchw', routeing, weight_real)
        f_imag = torch.einsum('bfc,fhw->bchw', routeing, weight_imag)
        
        # 组合回复数
        weight = torch.complex(f_real, f_imag)
        
        # --- E. 频域滤波 & iFFT ---
        x_fft = x_fft * weight
        x_out = torch.fft.irfft2(x_fft, s=(H, W), dim=(2, 3), norm='ortho')
        
        # --- F. 恢复原始精度 ---
        if x_out.dtype != original_dtype:
            x_out = x_out.to(original_dtype)
        
        return x_out
    
class ParallelSpectralRGBlock(nn.Module):
    """
    方案 A: 并行残差分支模块
    结构: Output = RGBlock(x) + alpha * FFT(x)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        
        # 1. 主干: 原汁原味的 RGBlock (负责局部特征 + 基础非线性)
        # 这样可以最大程度利用 Mamba-YOLO 的预训练权重
        self.main_branch = RGBlock(
            in_features, hidden_features, out_features, 
            act_layer=act_layer, drop=drop
        )
        
        # 2. 旁路: 频域分支 (负责全局纹理 + 去噪)
        # size=16 是一个经验值，适配常见特征图大小，支持插值
        self.spectral_branch = RobustSpectralBranch(in_features, size=16)
        
        # 3. 零初始化可学习系数 (Zero-initialized Gate)
        # 初始化为 0，保证训练初始阶段网络行为与原版完全一致
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 计算主干
        main_out = self.main_branch(x)
        
        # 计算频域分支
        spec_out = self.spectral_branch(x)
        
        # 融合: 原特征 + (权重 * 频域特征)
        # 随着 alpha 学习，频域信息会逐渐注入
        return main_out + self.alpha * spec_out    
       

#====================================================================

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
            self.mlp = ParallelSpectralRGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)
        self.post_attention = SimpleChannelAtt(hidden_dim, reduction=16)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        input = self.post_attention(input)
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
            self.mlp = ParallelSpectralRGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

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
