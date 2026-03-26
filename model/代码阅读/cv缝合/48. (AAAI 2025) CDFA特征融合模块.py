import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C3

'''
ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement
(AAAI 2025年 顶会)  
即插即用模块：CDFA 对比度驱动特征聚合模块（特征融合模块）
一、背景
1. 医学图像分割困境
a. 在医学图像领域，图像分割对于临床决策有着至关重要的意义。然而，医学图像的分割面临着许多挑战。
b. 医学图像中前景和背景的边界往往是模糊的，也就是所谓的 “软边界”。这种模糊的边界使得区分前景和背景变得困难。
c. 此外，医学图像的光照条件通常不好，而且对比度较低，这进一步增加了准确分割的难度。
d. 在医学图像中，共现现象非常普遍。这意味着模型在学习过程中很容易受到误导，学到一些不准确的特征，从而影响分割
的准确性。
e. 虽然现有的深度学习方法在一定程度上取得了进展，但这些方法在处理上述医学图像分割问题时，仍然存在不足。
2. CDFA的使命
CDFA是ConDSeg 框架中的一个关键模块。它的主要目的是利用医学图像中前景和背景的对比信息，引导多层次的特征融合，
并增强关键特征。通过这种方式，CDFA 能够克服共现现象对模型的干扰，提高模型区分不同实体的能力，进而提升医学图像分割的精度
和可靠性。

二、模块原理
1. 输入与初步处理
a. CDFA 模块首先接收来自 Encoder 输出的不同层次的特征图（从f1到f4）。
b. 这些特征图会先经过一组扩张卷积层进行预增强处理，经过处理后得到新的特征图（e1从到e4）。
c. 然后，把前一级的输出和侧向特征图（这里指的是e1、e2、e3）沿着通道方向进行拼接，形成一个新的特征图（e4直接输入到 CDFA 中）。
d. 与此同时，来自语义信息解耦（SID）模块的前景特征图和背景特征图会经过卷积层和双线性上采样操作，使它们的维度与匹配后，再输入到 CDFA 中。
2. 核心计算与融合机制
a. 当特征图输入到 CDFA 内部后，首先会使用多个 CBR 块（卷积 - 批量归一化 - 激活函数）对其进行初步融合。
b. 接着，通过一个线性层的权重，将特征图F映射成值向量V，并且把这个值向量在每个局部窗口展开。
c. 同时，前景特征图Ffg和背景特征图Fbg会经过两个不同的线性层，分别生成注意力权重Afg和Abg。
d. 在每个空间位置上，会根据这些注意力权重来计算注意力分数。具体操作是，先把前景和背景的注意力权重重塑，然后经过 Softmax 函数激活。
e. 之后，用激活后的注意力权重对展开的值向量进行两步加权操作。
f. 最后，把经过加权处理后的特征值进行密集聚合，这样就得到了最终的输出特征图，实现了多层次特征融合和关键特征增强。

三、适用任务：医学图像分割，目标检测，语义分割，图像分类，图像增强等所有计算机视觉CV任务通用的模块。
'''

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
"""Decouple Layer"""
class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        # f_uc = self.cbr_uc(x)
        # return f_fg, f_bg, f_uc
        return f_fg, f_bg

class CDFA(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, padding=1, stride=1,attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = out_c
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.dcp = DecoupleLayer(in_c,dim)
    def forward(self, x,fg, bg):

        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 实例化 CDFA模块
    cdfa = CDFA(in_c=64,out_c=64)
    x = torch.randn(1,64,32,32)
    fg = torch.randn(1, 64, 32, 32) #前景特征图
    bg = torch.randn(1, 64, 32, 32) #背景特征图
    # 这个模块前向传播输入张量 x, fg, 和 bg。
    output = cdfa(x,fg,bg)
    # 打印输出张量的形状
    print("input shape:", x.shape)
    print("Output shape:", output.shape)
