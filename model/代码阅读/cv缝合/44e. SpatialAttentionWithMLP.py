import torch
import torch.nn as nn

"""
创新点 5：空间注意力图的非线性变换 (Non-linear Transformation of Spatial Attention Map)
一、背景
传统的空间注意力图生成方法通常依赖于简单的卷积操作或池化方法，这些方法可能不足以捕捉到更复杂的空间模式和特征。
因此，简单的卷积和Sigmoid激活函数生成的空间注意力图可能会限制模型的表达能力，无法充分体现图像中的复杂空间结
构和语义信息。

二、实现方法
1. 引入多层感知机（MLP）：使用多层感知机（MLP）代替传统的简单卷积层来生成空间注意力图。MLP由多个全连接层构成，
每个层之间使用非线性激活函数（如ReLU、Leaky ReLU等），能够增强模型的非线性表达能力，从而更好地捕捉复杂的空间
模式。
2. 采用不同的激活函数：
在生成空间注意力图的过程中，除了使用ReLU激活函数外，还可以引入其他激活函数如Leaky ReLU、ELU等，以进一步增强
空间注意力图对不同空间模式的响应能力。
3. 非线性变换：
在空间注意力图生成过程中加入非线性变换，通过多层感知机或其他非线性结构，将传统的线性池化操作扩展为更强大的非线性
映射，使得模型能够学习到更复杂的空间关系和特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionWithMLP(nn.Module):
    def __init__(self, in_channels):
        """
        增强版空间注意力模块，通过MLP进行非线性变换增强空间注意力图的表达能力
        :param in_channels: 输入的通道数
        """
        super(SpatialAttentionWithMLP, self).__init__()

        # 空间注意力卷积层
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3)

        # 多层感知机（MLP）
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 32 * 32, in_channels * 8),  # 扁平化后的输入尺寸
            nn.ReLU(),
            nn.Linear(in_channels * 8, 1)
        )

        # Sigmoid激活生成空间注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播，生成带有非线性变换的空间注意力图
        :param x: 输入特征图
        :return: 生成的空间注意力图
        """
        batch_size, channels, height, width = x.size()

        # 生成空间注意力图（使用卷积操作）
        spatial_attention = self.conv1(x)

        # 使用MLP进行非线性变换
        x_flat = x.view(batch_size, -1)  # 扁平化为(B, C*H*W)，确保大小匹配
        mlp_out = self.mlp(x_flat)  # 输出大小为 (B, 1)

        # 将 MLP 输出进行扩展，使其与空间大小对齐
        mlp_out = mlp_out.view(batch_size, 1, 1, 1)  # (B, 1, 1, 1)
        mlp_out = mlp_out.expand(-1, 1, height, width)  # 扩展为 (B, 1, H, W)

        # 将卷积和MLP的结果相加，进行融合
        spatial_attention = spatial_attention + mlp_out

        # 使用sigmoid激活生成注意力图
        attention_map = self.sigmoid(spatial_attention)

        # 按照注意力图加权输入
        output = x * attention_map
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
    spatial_attention_mlp = SpatialAttentionWithMLP(in_channels=64)
    output_tensor = spatial_attention_mlp(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

