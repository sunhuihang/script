import torch
from torch import nn


# 所属机构：华为诺亚方舟实验室
"""
CV缝合救星魔改创新 72a. DynamicRepBN
1. DynamicRepBN类：在原RepBN基础上添加了scale_factor参数，用于控制动态权重调整的幅度。
在forward方法中，先计算输入特征的方差var，然后通过sigmoid函数将scale_factor与var的乘
积转换为权重bn_weight，这个权重在BatchNorm的输出和alpha * x之间动态分配，使得模型能根
据输入特征的方差情况自适应调整两者的贡献。
2. DynamicRepBN2d类：是针对四维数据（图像数据）的扩展，同样实现了根据输入数据方差动态调
整权重的功能，在forward方法中，计算方差时考虑了图像数据的所有空间维度（高度和宽度）。
3.主程序部分：创建了DynamicRepBN2d模型实例，并对随机生成的输入数据进行前向传播，打印模型
结构和输入输出数据的形状，以验证模型的正确性和可运行性。
"""


class DynamicRepBN(nn.Module):
    def __init__(self, channels):
        super(DynamicRepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)
        self.scale_factor = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = x.transpose(1, 2)
        var = torch.var(x, dim=(0, 2), keepdim=True)
        bn_weight = torch.sigmoid(self.scale_factor * var)
        x = bn_weight * self.bn(x) + (1 - bn_weight) * self.alpha * x
        x = x.transpose(1, 2)
        return x


# 可扩展到四维, 哔哩哔哩:CV缝合救星
class DynamicRepBN2d(nn.Module):
    def __init__(self, channels):
        super(DynamicRepBN2d, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)
        self.scale_factor = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        var = torch.var(x, dim=(0, 2, 3), keepdim=True)
        bn_weight = torch.sigmoid(self.scale_factor * var)
        x = bn_weight * self.bn(x) + (1 - bn_weight) * self.alpha * x
        return x


if __name__ == "__main__":
    # 模块参数
    batch_size = 1  # 批大小
    channels = 32  # 输入特征通道数
    height = 256  # 图像高度
    width = 256  # 图像宽度

    model = DynamicRepBN2d(channels=channels)
    print(model)
    print("哔哩哔哩:CV缝合救星, NB!")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = model(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)