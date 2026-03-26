# -*- coding: utf-8 -*-
# 环境: PyTorch >= 1.10

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 基础 ==========

class MyLayerNorm(nn.Module):
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
    return nn.Conv2d(in_ch, out_ch, k, s, k//2, bias=bias)

class PConv(nn.Module):
    """Partial Conv: 卷部分通道，降低开销"""
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
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.k = k
        self.base = nn.Conv2d(in_ch, out_ch, k, 1, k//2, bias=True)
        self.conv_offset = nn.Sequential(PConv(in_ch), nn.Conv2d(in_ch, 2*(k**2), 1, bias=True))
        self.conv_mask   = nn.Sequential(PConv(in_ch), nn.Conv2d(in_ch,   (k**2), 1, bias=True))
        nn.init.kaiming_uniform_(self.base.weight, a=math.sqrt(5))
        nn.init.zeros_(self.base.bias)
        # 让初始 mask 偏向1（门开）
        nn.init.constant_(self.conv_mask[-1].weight, 0.0)
        nn.init.constant_(self.conv_mask[-1].bias,   1.0)
    def forward(self, x):
        _ = self.conv_offset(x)          # 结构保留，参与学习
        m = torch.sigmoid(self.conv_mask(x))
        y = self.base(x)
        # 将mask压缩到与y通道一致：均值再扩展
        m_mean = m.mean(dim=1, keepdim=True)
        if m_mean.shape[1] != y.shape[1]:
            m_mean = m_mean.expand(-1, y.shape[1], -1, -1)
        return y * m_mean

# ========== SEDC ==========

class BlockGatingUnit(nn.Module):
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
    """SEDC 主体：LN->Spatial + 残差 -> LN->ChannelMLP + 残差"""
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

# ========== SADA（多头多膨胀 LSA） ==========

class LocalWindowExtractor(nn.Module):
    def __init__(self, k=3, dilation=1):
        super().__init__()
        self.k = k
        self.d = dilation
        self.pad = k//2 * dilation
    def forward(self, x):  # (B, C, H, W)
        # 输出: (B, C*k*k, H*W)
        return F.unfold(x, kernel_size=self.k, dilation=self.d, padding=self.pad, stride=1)

class SADA_Block(nn.Module):
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

            # 关键：k_win -> (B, H*W, dh, ks)，einsum 使用 'bdn,bndk->bnk'
            k_win = k_win.view(B, self.dh, -1, H*W).permute(0,3,1,2)  # (B,H*W,dh,ks)
            attn  = torch.einsum('bdn,bndk->bnk', qi_flat, k_win)     # (B,H*W,ks)
            attn  = F.softmax(attn, dim=-1)
            attn  = self.attn_drop(attn)

            v_win = v_win.view(B, self.dh, -1, H*W).permute(0,3,2,1)  # (B,H*W,ks,dh)
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

# ========== SADTUnit：单尺度即插即用模块 ==========

class SADT(nn.Module):
    """
    单尺度 SADT 模块（即插即用）:
      - 接口对齐 GCConv: SADTUnit(in_ch, out_ch, kernel_size=3, stride=1, depth=4, num_heads=4)
      - 结构: 1x1 投影 -> SEDC -> (SADA * (depth-2)) -> SEDC -> 1x1 投影
      - stride>1 时在输入侧做 stride 卷积投影，实现可选下采样
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, depth=4, num_heads=4, win=8):
        super().__init__()
        assert depth >= 2, "depth 至少为 2（首尾各一层 SEDC）"
        mid_ch = out_ch  # 把中间通道定为 out_ch，便于对接
        self.kernel_size = kernel_size
        self.stride = stride
        # 输入投影（含 stride）
        self.proj_in  = nn.Conv2d(in_ch, mid_ch, 1, stride=stride, padding=0, bias=True)
        # 首层 SEDC
        self.first = ScaleEnhancedDC(mid_ch, block=win)
        # 中间 SADA 堆叠
        self.blocks = nn.ModuleList([
            SADA_Block(dim=mid_ch, num_heads=num_heads, k=3, ms_list=(1,3,5,7))
            for _ in range(max(0, depth-2))
        ])
        # 末层 SEDC
        self.last  = ScaleEnhancedDC(mid_ch, block=win)
        # 输出投影
        self.proj_out = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.proj_in(x)
        s = x
        x = self.first(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.last(x) + s
        x = self.proj_out(x)
        return x

    def __repr__(self):
        return f"SADTUnit(in_ch=?, out_ch=?, k={self.kernel_size}, s={self.stride})"


if __name__ == "__main__":
    input = torch.rand(1, 64, 32, 32)
    GCConv1 = SADT(64, 64, kernel_size=3, stride=1, depth=4, num_heads=4)  # 只特征提取，不下采样
    print(GCConv1)
    output = GCConv1(input)
    print('CV缝合救星即插即用模块永久更新-GCConv input_size:',  input.size())
    print('CV缝合救星即插即用模块永久更新-GCConv output_size:', output.size())
