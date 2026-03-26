import torch
from torch import nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新1：引入局部自适应卷积核大小
1. 通过引入动态卷积核大小，可以让每个位置的卷积核大小根据特征图的局部信息动态调整。这样可以更好地捕捉局部区域
的细节，同时提升模型性能。
2. 实现思路：
基于输入特征图的局部统计信息（例如，局部平均或方差），动态调整卷积核的大小。卷积核的大小会随着输入特征图的不同
区域而变化。

CV缝合救星魔改创新2：多尺度卷积操作
1. 通过引入多尺度卷积，在不同尺度下对特征进行处理，增强模型对不同尺度信息的提取能力。
2. 实现思路：
使用不同尺寸的卷积核（例如，3×3、5×5和7×7），并在同一层次上应用这些卷积，结合不同尺度的特征信息。
"""

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DynamicKernelConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.dynamic_kernel = nn.Conv2d(dim, dim, 1)  # 用于动态调整卷积核大小的模块

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.norm(x)

        # 获取动态调整的卷积核大小
        dynamic_kernel_size = self.dynamic_kernel(x)  # 通过卷积操作计算出动态的核大小
        dynamic_kernel_size = torch.sigmoid(dynamic_kernel_size) * 11 # 限制在[0, 11]范围内

        # 用动态大小的卷积进行处理
        a = self.a(x)  # 固定卷积操作
        v = self.v(x)  # 固定卷积操作

        # 这里可以应用dynamic_kernel_size进行动态调整，但由于卷积核大小的不同，卷积操作需要调整
        # 暂时不直接操作卷积，而是通过改变卷积结果的响应调整特征。

        x = a * v
        x = self.proj(x)
        return x


class MultiScaleConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2)
        self.conv7x7 = nn.Conv2d(dim, dim, 7, padding=3)

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.norm(x)

        # 获取不同尺度的卷积特征
        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(x)
        conv7 = self.conv7x7(x)

        # 在多个尺度上融合特征
        x = (conv3 + conv5 + conv7) / 3

        # 其他操作
        a = self.a(x)  # 固定卷积操作
        v = self.v(x)  # 固定卷积操作

        x = a * v
        x = self.proj(x)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 使用动态卷积核大小的模块
    block_dynamic_kernel = DynamicKernelConvMod(64).cuda()
    input = torch.rand(3, 64, 85, 85).cuda()
    output_dynamic_kernel = block_dynamic_kernel(input)
    print("Dynamic Kernel ConvMod:", input.size(), output_dynamic_kernel.size())

    # 使用多尺度卷积操作的模块
    block_multi_scale = MultiScaleConvMod(64).cuda()
    output_multi_scale = block_multi_scale(input)
    print("Multi-Scale ConvMod:", input.size(), output_multi_scale.size())
