import torch
import torch.nn as nn

"""
创新1：多尺度卷积空间注意力
一、背景
传统的空间注意力通常使用单一大小的卷积核进行特征图处理，但特征图中的空间信息可能在不同尺度下有不同的表现。
多尺度卷积可以通过不同大小的卷积核捕捉不同尺度的空间信息。
二、实现方法
通过多个不同尺度的卷积（例如 3x3 和 5x5 卷积核）提取空间特征，并将它们结合起来生成最终的空间注意力图。
"""

class MultiScaleConvSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        """
        使用多尺度卷积生成空间注意力图
        :param in_channels: 输入的通道数
        """
        super(MultiScaleConvSpatialAttention, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        生成空间注意力图，并加权输入特征
        :param x: 输入特征图
        :return: 加权后的输出特征图
        """
        # 通过不同尺度的卷积生成不同的空间注意力图
        attention_map_3x3 = self.conv3x3(x)
        attention_map_5x5 = self.conv5x5(x)
        attention_map_7x7 = self.conv7x7(x)

        # 将所有注意力图进行融合
        attention_map = attention_map_3x3 + attention_map_5x5 + attention_map_7x7
        attention_map = self.sigmoid(attention_map)

        # 按照注意力图加权输入特征
        output = x * attention_map
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
    multi_scale_attention = MultiScaleConvSpatialAttention(in_channels=64)
    output_tensor = multi_scale_attention(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
