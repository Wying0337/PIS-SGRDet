import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math
import numpy as np


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8
        
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        
        x=x[:,:,:h,:w]
        
        return x   
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SpectralRGBlock(nn.Module):
    """
    Fused Block combining RGBlock (Local Spatial) and EDFFN (Patch-based Spectral).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., patch_size=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 通常在 GLU 结构中，隐藏层维度会稍微收缩以保持效率，或者扩张
        # 这里参考 RGBlock 的设计: hidden_features = int(2 * hidden_features / 3)
        # 但为了容纳双分支，我们保持 hidden_features 与输入一致或由外部传入
        self.mid_features = int(2 * hidden_features / 3) 
        
        # 1. Projection Input (Expands to 2 * mid_features for GLU split)
        self.fc1 = nn.Conv2d(in_features, self.mid_features * 2, kernel_size=1)

        # 2. Spatial Branch (From RGBlock: DWConv)
        self.dwconv = nn.Conv2d(self.mid_features, self.mid_features, kernel_size=3, 
                                stride=1, padding=1, bias=True, groups=self.mid_features)

        # 3. Spectral Branch (From EDFFN: Patch-wise FFT)
        self.patch_size = patch_size
        # FFT Parameter: shape [C, 1, 1, H_patch, W_patch_half]
        # output of rfft2 on 8x8 patch has last dim size of 8//2 + 1 = 5
        self.fft_filter = nn.Parameter(
            torch.ones((self.mid_features, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )

        self.act = act_layer()
        self.fc2 = nn.Conv2d(self.mid_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 1. Expand and Split (GLU style)
        # x_path: 用于特征处理的主路径
        # v_gate: 用于门控的路径
        x_path, v_gate = self.fc1(x).chunk(2, dim=1)

        # 2. Spatial Processing (Residual DWConv)
        # RGBlock logic: x = dwconv(x) + x
        x_spatial = self.dwconv(x_path) + x_path

        # 3. Spectral Processing (EDFFN logic inserted here)
        B, C, H, W = x_path.shape
        
        # Padding to multiple of patch_size
        h_n = (self.patch_size - H % self.patch_size) % self.patch_size
        w_n = (self.patch_size - W % self.patch_size) % self.patch_size
        
        x_padded = F.pad(x_path, (0, w_n, 0, h_n), mode='reflect')
        
        # Rearrange to patches: [B, C, H_grid, W_grid, pH, pW]
        x_patches = rearrange(x_padded, 'b c (h p1) (w p2) -> b c h w p1 p2', 
                              p1=self.patch_size, p2=self.patch_size)
        
        # FFT
        x_fft = torch.fft.rfft2(x_patches.float())
        
        # Spectral Gating/Filtering
        x_fft = x_fft * self.fft_filter
        
        # Inverse FFT
        x_spectral_patches = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        
        # Restore spatial structure
        x_spectral = rearrange(x_spectral_patches, 'b c h w p1 p2 -> b c (h p1) (w p2)', 
                               p1=self.patch_size, p2=self.patch_size)
        
        # Crop back to original size
        x_spectral = x_spectral[:, :, :H, :W]

        # 4. Fusion
        # 将空间局部特征与频域特征融合。
        # 简单的相加在梯度流上通常是最稳定的。
        x_fused = x_spatial + x_spectral

        # 5. Activation & Gating (GLU)
        x = self.act(x_fused) * v_gate
        
        # 6. Output Projection
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
    


    #----------------------------------------------------------------------------
class SRU(nn.Module):
    """ Spatial Reconstruction Unit: 抑制背景噪声 """
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=2, num_channels=channels, eps=1e-5)
        # 用 Sigmoid 生成空间注意力 mask
        # 这里的实现简化了原始论文，利用 GN 的统计特性来区分前景背景
    
    def forward(self, x):
        b, c, h, w = x.size()
        # GroupNorm 计算均值方差，作为内容描述符
        gn_x = self.group_norm(x)
        w_gamma = self.group_norm.weight.reshape(1, c, 1, 1)
        b_beta = self.group_norm.bias.reshape(1, c, 1, 1)
        
        # 增强后的特征
        # 原始论文使用了更复杂的门控，这里使用 GN 权重的简易实现
        # 核心思想：利用 GN 的 affine 参数来判断通道重要性，利用 x 本身判断空间重要性
        gn_x = gn_x * w_gamma + b_beta
        
        # 生成 Mask (0~1)
        # 激活值大的认为是前景(种子)，小的认为是背景
        attn = torch.sigmoid(gn_x)
        
        # 门控机制：分开处理
        x_1 = x * attn      # 保留信息量大的部分
        x_2 = x * (1 - attn)# 背景部分
        
        # 将背景部分的信息"重建"融合回 x_1 (通过 1x1 卷积，这里简化为直接相加)
        # 实际代码中，SRU 输出的是重构后的特征
        return x_1 + x_2 * 0.5 # 简单加权，抑制背景

class CRU(nn.Module):
    """ Channel Reconstruction Unit: 精炼通道信息 """
    def __init__(self, op_channel, alpha=1/2, squeeze_radio=2, group_size=2, group_kernel_size=3):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        
        # Up path: Global info
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)
        
        # Low path: Local info
        self.PWC1 = nn.Conv2d(low_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        
        # Transform
        up = self.squeeze1(up)
        low = self.squeeze2(low)
        
        # Reconstruct
        # 上分支看大尺度，下分支看小尺度
        Y1 = self.GWC(up) + self.PWC1(low)
        
        return Y1

class ScRGBlock(nn.Module):
    """
    结合 ScConv 的 RGBlock (CVPR 2023) - 修复版
    修复了 hidden_features 为奇数导致 GroupNorm 报错的问题
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 计算隐藏层维度
        hidden_features = int(2 * hidden_features / 3)
        
        # === [Fix: 关键修复] ===
        # GroupNorm(num_groups=2) 要求通道数必须能被 2 整除
        # 如果是奇数，我们强制 +1 变成偶数
        if hidden_features % 2 != 0:
            hidden_features += 1
        # ======================
        
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        
        # 使用 SRU (空间重建) 替代普通 DWConv
        self.sru = SRU(hidden_features)
        
        # 注意：这里 groups=hidden_features，所以 hidden_features 必须 > 0
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x
        
        # Expand
        x_expanded = self.fc1(x)
        x, v = x_expanded.chunk(2, dim=1)
        
        # 1. 空间去噪 (SRU)
        x_clean = self.sru(x)
        
        # 2. 特征提取
        x = self.act(self.dwconv(x_clean) + x_clean) * v
        
        # Project Out
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x + shortcut
    
 #----------------------------------------------------------------------------
class HaarWaveletDownsampling(nn.Module):
    """
    基于 Haar 小波的下采样模块 (CVPR 2024 思想简化版)
    将输入分解为: 低频(LL), 水平高频(LH), 垂直高频(HL), 对角高频(HH)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # Haar 小波核，不可学习，固定参数
        self.register_buffer('haar_weights', torch.eye(4).view(4, 1, 2, 2) / 2)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # 重塑以进行 2x2 块操作
        # 这种写法比 F.conv2d 更快且无 padding 问题
        x_reshaped = x.view(b, c, h // 2, 2, w // 2, 2)
        x_reshaped = x_reshaped.permute(0, 1, 3, 5, 2, 4).contiguous() # [B, C, 2, 2, H/2, W/2]
        x_reshaped = x_reshaped.view(b, c * 4, h // 2, w // 2)
        
        # 此时通道顺序是: 
        # C0_TL, C0_TR, C0_BL, C0_BR, C1_TL... (TL=TopLeft)
        # 我们需要将其转换为 Haar 分量: LL, LH, HL, HH
        
        # 为了简单高效，我们直接使用加减法实现 Haar
        # x00 = TopLeft, x01 = TopRight, x10 = BottomLeft, x11 = BottomRight
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # LL: 平均值 (低频背景)
        ll = (x00 + x01 + x10 + x11) / 4
        # LH: 水平边缘 (根的横向特征)
        lh = (x00 - x01 + x10 - x11) / 4
        # HL: 垂直边缘 (根的纵向特征)
        hl = (x00 + x01 - x10 - x11) / 4
        # HH: 对角边缘 (根的交叉特征)
        hh = (x00 - x01 - x10 + x11) / 4
        
        # 拼接: [B, 4*C, H/2, W/2]
        return torch.cat([ll, lh, hl, hh], dim=1)

class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        # 1. 小波下采样 (通道数变 4 倍)
        self.wt_down = HaarWaveletDownsampling(inp)
        self.in_features = inp * 4
        
        self.hidden_dims = embed_dim // 2
        
        # 2. 特征融合卷积
        # 处理小波分解后的高频信息，提取边缘
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_features, self.hidden_dims, kernel_size=ks, padding=ks//2, bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU()
        )
        
        # 3. 二次下采样 (普通卷积)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=ks//2, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU()
        )
        def forward(self, x):
            x = self.wt_down(x) # [B, 12, H/2, W/2] (假设inp=3)
            x = self.conv1(x)   # 融合高频边缘
            x = self.conv2(x)   # 再下采样
            return x
#==============================================================================================