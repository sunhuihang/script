import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C3

"""
CV缝合救星魔改创新1: 可调节的对比度增强层
一、背景：
在医学图像分割等任务中，图像的前景（通常是目标区域，如肿瘤、器官等）和背景之间的对比度可能较低，这使得模型很难区分它们。
对比度增强的作用就是提升图像中前景和背景之间的差异，使得模型可以更好地提取和区分目标区域。这对于提高分割任务的准确性和
效果非常重要。
二、创新点：
1. 对比度增强：通过 contrast_enhance 层，增加了对输入特征图的对比度增强。这将有助于提升前景与背景之间的对比，使得分割任务的效果更加突出。
通过卷积和激活函数提升输入图像的显著特征。
2. 对比度增强层：在 CDFA 模块的 forward 方法中，首先使用对比度增强层对输入进行处理，从而让模型能更好地处理低对比度的医学图像数据。
"""
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
        return f_fg, f_bg


class CDFA(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, padding=1, stride=1, attn_drop=0., proj_drop=0.):
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
        self.dcp = DecoupleLayer(in_c, dim)

        # 可调节的对比度增强层
        self.contrast_enhance = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, fg, bg):
        # 1. 对输入进行对比度增强
        x = self.contrast_enhance(x)

        # 2. 输入前景和背景特征图
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)

        # 3. 计算前景的注意力
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        # 4. 计算背景的注意力
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
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding,
                            stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 实例化 CDFA模块
    cdfa = CDFA(in_c=64, out_c=64)
    x = torch.randn(1, 64, 32, 32)
    fg = torch.randn(1, 64, 32, 32)  # 前景特征图
    bg = torch.randn(1, 64, 32, 32)  # 背景特征图
    # 这个模块前向传播输入张量 x, fg, 和 bg。
    output = cdfa(x, fg, bg)
    # 打印输出张量的形状
    print("input shape:", x.shape)
    print("Output shape:", output.shape)
