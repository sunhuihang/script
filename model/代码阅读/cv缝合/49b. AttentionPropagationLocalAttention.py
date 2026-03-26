import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CV缝合救星魔改创新2：AttentionPropagationLocalAttention注意力传播机制
一、原理
1. 在传统的注意力机制中，信息的传递往往局限于局部或单一的计算流程，缺乏对特征之间更广泛传播和交互的有效利用。
注意力传播机制旨在打破这种局限，通过在特征图上建立一种信息传播路径，使得局部计算得到的注意力信息能够在一定范
围内扩散和影响周围的特征。
2. 在图像的不同区域中，某些局部区域可能具有显著的特征，但这些特征对于周边区域的影响可能在传统方法中未被充分
挖掘。注意力传播机制能够让这些显著区域的注意力信息像涟漪一样向周围传递，从而增强特征之间的关联性和整体性。在
超分辨率任务中，图像的细节信息可能分散在不同位置，通过注意力传播，可以将这些细节相关的注意力信息汇聚和共享，
有助于更全面地恢复图像的高频信息，提升重建效果。
二、实现方法
1. 建立传播连接
在原有的注意力模块基础上，添加额外的卷积层或全连接层来构建传播路径。例如，可以在计算出局部注意力权重后，使用一
个可学习的卷积层，其卷积核的大小和步长设置为能够在一定程度上扩大感受野的数值，如 3x3 的卷积核，步长为 1，对加
权后的特征图进行处理，实现注意力信息的初步传播。
2. 信息融合与更新
在传播过程中，需要设计合适的融合机制。一种方式是将传播后的特征与原始特征进行加权相加或拼接后再经过一个激活函数进
行处理。在得到传播后的特征图后，将其与原始输入特征图按照一定比例进行加权相加，比例系数可以是可学习的参数，通过反
向传播进行优化。然后再经过一个 Sigmoid 或 ReLU 等激活函数，更新特征图，使得传播后的注意力信息能够更好地融入到
整体特征表示中，增强特征的表达能力和注意力机制的效果。
"""

class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class AttentionPropagationLocalAttention(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, f, 1)
        self.softpool = SoftPooling2D(7, stride=3)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(f, channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        # 新增注意力传播卷积层
        self.propagate_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # 计算局部重要性
        x1 = self.conv1(x)
        x2 = self.softpool(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        w = self.sigmoid(x4)

        # 调整 w 的尺寸使其与 x 匹配
        w = F.interpolate(w, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # 注意力传播
        x_propagated = self.propagate_conv(x * w)

        # 门控机制
        g = self.sigmoid(x[:, :1].clone())

        return x * w * g + x_propagated


if __name__ == "__main__":
    input = torch.randn(1, 32, 64, 64)
    ALA = AttentionPropagationLocalAttention(32)
    output = ALA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())