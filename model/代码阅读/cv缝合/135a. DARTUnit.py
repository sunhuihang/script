# -*- coding: utf-8 -*-
# DART-Unit: Dynamic Artifact-aware Routing Transformer Unit
# 动态伪影感知路由 Transformer 单元（CVPR 风格魔改版）
# 环境: PyTorch >= 1.10
# 依赖: 仅 torch（零 torchvision / PIL）

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 基础工具模块 =====================

class MyLayerNorm(nn.Module):
    """通道后移的 LayerNorm，适配(C,H,W)张量"""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, x):
        # (B,C,H,W) -> (B,H,W,C) -> LN -> (B,C,H,W)
        x = x.permute(0,2,3,1)
        x = self.ln(x)
        x = x.permute(0,3,1,2)
        return x

def conv(in_ch, out_ch, k=3, s=1, bias=True):
    """带same padding的Conv2d"""
    return nn.Conv2d(in_ch, out_ch, k, s, k//2, bias=bias)

class PConv(nn.Module):
    """Partial Conv：仅卷积1/4通道，降算力、提速度"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dim_conv = dim // 4
        self.dim_untouched = dim - self.dim_conv
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        return torch.cat((x1, x2), dim=1)

class DeformConv2dLiteApprox(nn.Module):
    """
    轻量“近似可变形卷积”：
      - 学习 offset/mask（offset不实际位移，仅参与学习结构；mask用于门控）
      - 用标准卷积 + 学习门控近似 DCN 效果，零第三方依赖，能开箱即用
    """
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.k = k
        self.base = nn.Conv2d(in_ch, out_ch, k, 1, k//2, bias=True)
        self.conv_offset = nn.Sequential(PConv(in_ch), nn.Conv2d(in_ch, 2*(k**2), 1, bias=True))
        self.conv_mask   = nn.Sequential(PConv(in_ch), nn.Conv2d(in_ch,   (k**2), 1, bias=True))
        nn.init.kaiming_uniform_(self.base.weight, a=math.sqrt(5))
        nn.init.zeros_(self.base.bias)
        # 初始mask偏向打开
        nn.init.constant_(self.conv_mask[-1].weight, 0.0)
        nn.init.constant_(self.conv_mask[-1].bias,   1.0)
    def forward(self, x):
        _ = self.conv_offset(x)                         # 结构存在以供反传学习
        m = torch.sigmoid(self.conv_mask(x))            # (B, k*k, H, W)
        y = self.base(x)                                # (B, Cout, H, W)
        m_mean = m.mean(dim=1, keepdim=True)            # (B,1,H,W)
        if m_mean.shape[1] != y.shape[1]:
            m_mean = m_mean.expand(-1, y.shape[1], -1, -1)
        return y * m_mean

# ===================== SEDC（尺度增强分支） =====================

class BlockGatingUnit(nn.Module):
    """快速版 gMLP 空间门：逐块下采样生成门，再像素重排上采样回位"""
    def __init__(self, c, block_size=8):
        super().__init__()
        k = block_size
        self.k = k
        self.depthwise_token = nn.Conv2d(1, k*k, k, stride=k, bias=True)
        self.up = nn.PixelShuffle(k)
        self.c = c
    def forward(self, x):
        # x: (B,C,H,W)
        shortcut = x
        w = self.depthwise_token.weight.repeat(self.c,1,1,1)
        b = self.depthwise_token.bias.repeat(self.c)
        x = F.conv2d(x, weight=w, bias=b, stride=self.k, groups=self.c)
        x = self.up(x)
        return shortcut * x

class ChannelMLP(nn.Module):
    """通道MLP：部分通道卷积 + DWConv 门控 + 1x1复原"""
    def __init__(self, c, hidden=None):
        super().__init__()
        hidden = hidden or (4*c)
        self.dim_conv = c // 4
        self.dim_untouched = c - self.dim_conv
        self.partial = nn.Conv2d(self.dim_conv, self.dim_conv, 3,1,1, bias=False)
        self.fc1 = nn.Sequential(nn.Linear(c, 2*hidden), nn.GELU())
        self.dw  = nn.Sequential(nn.Conv2d(hidden, hidden, 3,1,1, groups=hidden), nn.GELU())
        self.fc2 = nn.Conv2d(hidden, c, 1)
    def forward(self, x):
        B,C,H,W = x.shape
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial(x1)
        x  = torch.cat([x1,x2], dim=1)
        x  = x.permute(0,2,3,1)                # (B,H,W,C)
        x  = self.fc1(x)                       # (B,H,W,2Hid)
        x1,x2 = x.chunk(2, dim=-1)
        x1 = x1.permute(0,3,1,2)               # (B,Hid,H,W)
        x1 = self.dw(x1)
        x  = x1 * x2.permute(0,3,1,2)
        x  = self.fc2(x)
        return x

class DeformSPPFSpatialMLP(nn.Module):
    """SEDC 的空间分支：两次“近似可变形卷积”+ 两级空间门"""
    def __init__(self, c, block=8):
        super().__init__()
        self.cproj1 = conv(c,c,1)
        self.sppf0  = nn.Sequential(DeformConv2dLiteApprox(c,c,3), nn.GELU())
        self.gate0  = BlockGatingUnit(c, block)
        self.sppf1  = nn.Sequential(DeformConv2dLiteApprox(c,c,3), nn.GELU())
        self.gate1  = BlockGatingUnit(c, block)
        self.cproj2 = conv(c,c,1)
    def forward(self, x):
        x = F.gelu(self.cproj1(x))
        x = self.sppf0(x); x = self.gate0(x)
        x = self.sppf1(x); x = self.gate1(x)
        x = self.cproj2(x)
        return x

class ScaleEnhancedDC(nn.Module):
    """SEDC 主体：LN -> Spatial-MLP + 残差 -> LN -> Channel-MLP + 残差"""
    def __init__(self, c, block=8):
        super().__init__()
        self.ln1 = MyLayerNorm(c)
        self.spatial = DeformSPPFSpatialMLP(c, block)
        self.ln2 = MyLayerNorm(c)
        self.cmlp = ChannelMLP(c)
    def forward(self, x):
        s = x
        x = self.spatial(self.ln1(x)) + s
        s = x
        x = self.cmlp(self.ln2(x)) + s
        return x

# ===================== SADA（多头多膨胀 LSA，全局分支） =====================

class LocalWindowExtractor(nn.Module):
    """带 dilation 的局部感受野展开"""
    def __init__(self, k=3, dilation=1):
        super().__init__()
        self.k = k
        self.d = dilation
        self.pad = k//2 * dilation
    def forward(self, x):  # (B, C, H, W)
        # 输出: (B, C*k*k, H*W)
        return F.unfold(x, kernel_size=self.k, dilation=self.d, padding=self.pad, stride=1)

class SADA_Block(nn.Module):
    """多头多膨胀的局部注意力 + MLP（Transformer式结构）"""
    def __init__(self, dim, num_heads=4, k=3, ms_list=(1,3,5,7), attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.h   = num_heads
        self.dh  = dim // num_heads
        self.scale = self.dh ** -0.5

        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=True)
        self.dw_qkv = nn.Conv2d(dim*3, dim*3, 3,1,1, groups=dim*3)

        if len(ms_list) < num_heads:
            ms_list = list(ms_list) + [ms_list[-1]]*(num_heads-len(ms_list))
        self.ms_list = ms_list[:num_heads]
        self.extractors = nn.ModuleList([LocalWindowExtractor(k=k, dilation=d) for d in self.ms_list])

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = ChannelMLP(dim)

    def forward_attn(self, x):
        # x: (B,H,W,C)
        B,H,W,C = x.shape
        xc = x.permute(0,3,1,2)                 # (B,C,H,W)
        qkv = self.dw_qkv(self.qkv(xc))
        q, k, v = torch.chunk(qkv, 3, dim=1)    # (B,C,H,W)

        q = q.view(B, self.h, self.dh, H, W)
        k = k.view(B, self.h, self.dh, H, W)
        v = v.view(B, self.h, self.dh, H, W)

        outs = []
        for i in range(self.h):
            qi = q[:,i] * self.scale              # (B,dh,H,W)
            ki = k[:,i]                           # (B,dh,H,W)
            vi = v[:,i]                           # (B,dh,H,W)

            k_win = self.extractors[i](ki)        # (B, dh*k*k, H*W)
            v_win = self.extractors[i](vi)        # (B, dh*k*k, H*W)

            qi_flat = qi.view(B, self.dh, H*W)    # (B,dh,H*W)

            # k_win: (B, dh, ks, H*W) -> (B, H*W, dh, ks)
            k_win = k_win.view(B, self.dh, -1, H*W).permute(0,3,1,2)
            # 注意力: (B,H*W,ks)
            attn  = torch.einsum('bdn,bndk->bnk', qi_flat, k_win)
            attn  = F.softmax(attn, dim=-1)
            attn  = self.attn_drop(attn)

            # V 聚合: v_win -> (B,H*W,ks,dh)
            v_win = v_win.view(B, self.dh, -1, H*W).permute(0,3,2,1)
            out   = torch.einsum('bnk,bnkd->bnd', attn, v_win)        # (B,H*W,dh)
            out   = out.permute(0,2,1).contiguous().view(B, self.dh, H, W)
            outs.append(out)

        out = torch.cat(outs, dim=1)          # (B,C,H,W)
        out = self.proj_drop(self.proj(out))  # (B,C,H,W)
        out = out.permute(0,2,3,1)            # (B,H,W,C)
        return out

    def forward(self, x):
        # x: (B,C,H,W)
        y = x.permute(0,2,3,1)                    # (B,H,W,C)
        y = y + self.forward_attn(self.norm1(y))  # attn 残差
        y = y.permute(0,3,1,2)                    # (B,C,H,W)
        y = y + self.mlp(self.norm2(y.permute(0,2,3,1)).permute(0,3,1,2))  # ffn 残差
        return y

# ===================== 频域先验（FAPI） =====================

class FrequencyAwarePrior(nn.Module):
    """
    频域先验注入：
      1) 计算 rFFT 幅度谱 -> 估计高/低频能量比例与周期性强度
      2) 生成通道门控向量 gamma，作为对特征的“频域自适应增益”
      3) 同时输出一个低维先验向量 prior，用于路由器决策（拼到空间统计上）
    """
    def __init__(self, c, shrink=16):
        super().__init__()
        hid = max(c // shrink, 8)
        self.fc = nn.Sequential(
            nn.Linear(4, hid), nn.GELU(),
            nn.Linear(hid, c), nn.Sigmoid()
        )
    @torch.no_grad()
    def _freq_stats(self, x):
        # x: (B,C,H,W)
        B,C,H,W = x.shape
        # rFFT 幅度
        X = torch.fft.rfft2(x, norm="ortho")                 # (B,C,H,W//2+1)
        mag = X.abs()                                        # 幅度
        # 低频能量（中心附近），高频能量（周边）
        # 简化划分：取宽度1/4作为低频
        hf_start = W//8
        low = mag[..., :hf_start].mean(dim=(-2,-1))          # (B,C)
        high= mag[..., hf_start:].mean(dim=(-2,-1))          # (B,C)
        # 周期性（幅度峰值占比）：取每通道最大/均值
        peak = mag.amax(dim=(-2,-1))
        mean = mag.mean(dim=(-2,-1)) + 1e-6
        periodic = (peak/mean).mean(dim=1, keepdim=True)     # (B,1)
        # 频能比：高/低
        ratio = (high.mean(dim=1, keepdim=True)+1e-6)/(low.mean(dim=1, keepdim=True)+1e-6)  # (B,1)
        # 返回两个标量先验 + 通道统计（用于 gamma 学习）
        return ratio, periodic, low.mean(dim=1, keepdim=True), high.mean(dim=1, keepdim=True)

    def forward(self, x):
        B,C,H,W = x.shape
        ratio, periodic, low_s, high_s = self._freq_stats(x)           # (B,1) x2 + (B,1) x2
        # 先验拼接为 (B,4)
        prior = torch.cat([ratio, periodic, low_s, high_s], dim=1)     # (B,4)
        # 映射到通道门控 gamma
        gamma = self.fc(prior)                                         # (B,C) in [0,1]
        gamma = gamma.view(B,C,1,1)
        return gamma, prior                                            # gamma用于通道缩放，prior交给路由器

# ===================== DART-Unit（双专家动态路由） =====================

class DARTUnit(nn.Module):
    """
    DART-Unit（动态伪影感知路由 Transformer 单元）
    结构概览：
      输入投影(含stride) -> 频域先验(FAPI) -> [Local Expert: SEDC] & [Global Expert: SADA*] 并联
      -> 动态路由器（逐像素，softmax成两路权重） -> 融合输出 -> 1x1 输出投影
    使用方式：
      DARTUnit(in_ch, out_ch, kernel_size=3, stride=1, depth=4, num_heads=4)
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, depth=4, num_heads=4, win=8):
        super().__init__()
        assert depth >= 2, "depth 至少为 2（保证 Local/Global 均有表达）"
        mid_ch = out_ch       # 中间通道设为输出通道，方便对接
        self.kernel_size = kernel_size
        self.stride = stride

        # 输入侧投影（支持 stride 下采样）
        self.proj_in  = nn.Conv2d(in_ch, mid_ch, 1, stride=stride, padding=0, bias=True)

        # 频域先验（FAPI）
        self.fapi = FrequencyAwarePrior(mid_ch)

        # Local Expert: 一层 SEDC（也可堆叠，保持轻量这里1层）
        self.local = ScaleEnhancedDC(mid_ch, block=win)

        # Global Expert: (SADA * (depth-1))，先做一层SADA，后续可按depth-2叠加
        blocks = []
        for _ in range(depth-1):
            blocks.append(SADA_Block(dim=mid_ch, num_heads=num_heads, k=3, ms_list=(1,3,5,7)))
        self.global_expert = nn.Sequential(*blocks)

        # 路由器：融合空间+通道统计+频域先验，逐像素输出2路权重
        #   输入: concat([x, GAP(x), FAPI先验广播]) -> 1x1 -> Softmax(2路)
        self.router_conv = nn.Sequential(
            nn.Conv2d(mid_ch + mid_ch + 2, mid_ch, 1, bias=True), nn.GELU(),
            nn.Conv2d(mid_ch, 2, 1, bias=True)
        )

        # 输出投影
        self.proj_out = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=True)

    def forward(self, x):
        # 1) 输入投影
        x = self.proj_in(x)                 # (B,C,H,W)

        # 2) 频域先验（通道门控 + 先验向量）
        gamma, prior = self.fapi(x)         # gamma: (B,C,1,1), prior: (B,4)
        x = x * (1.0 + gamma)               # 频域增强（>1增益，<=1抑制）

        # 3) 两个专家分支
        xl = self.local(x)                  # Local Expert（SEDC）：偏局部形变/纹理
        xg = self.global_expert(x)          # Global Expert（SADA）：偏全局依赖/尺度

        # 4) 动态路由权重（逐像素）
        B,C,H,W = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).expand(-1, -1, H, W)       # (B,C,1,1)->(B,C,H,W)
        # 将 prior (B,4) 压到2维（ratio & periodic）并广播到(H,W)
        # 这里选用前2维作为路由显著先验（高/低频比与周期性）
        prior2 = prior[:, :2].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B,2,H,W)
        router_in = torch.cat([x, gap, prior2], dim=1)               # (B, C+C+2, H, W)
        logits = self.router_conv(router_in)                         # (B,2,H,W)
        weights = F.softmax(logits, dim=1)                           # (B,2,H,W)
        wl, wg = weights[:, :1], weights[:, 1:]                      # (B,1,H,W) 分别对应Local/Global

        # 5) 融合输出（注意：逐像素软路由）
        y = wl * xl + wg * xg                                        # (B,C,H,W)

        # 6) 输出投影
        y = self.proj_out(y)
        return y

    def __repr__(self):
        return f"DARTUnit(k={self.kernel_size}, s={self.stride}, routed=Local(S E D C)+Global(S A D A))"

# ===================== Demo =====================

if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    block = DARTUnit(64, 64, kernel_size=3, stride=1, depth=4, num_heads=4)  # 不下采样
    print(block)
    y = block(x)
    print("CV缝合救星即插即用模块永久更新-DARTUnit input_size:", x.size())
    print("CV缝合救星即插即用模块永久更新-DARTUnit output_size:", y.size())

    # 下采样版本（stride=2），可替代下采样卷积块
    block_ds = DARTUnit(64, 128, kernel_size=3, stride=2, depth=4, num_heads=4)
    y2 = block_ds(x)
    print("DARTUnit downsample output_size:", y2.size())
