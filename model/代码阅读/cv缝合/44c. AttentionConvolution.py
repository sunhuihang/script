import torch
import torch.nn as nn

"""
注意力卷积 (Attention Convolutions)
一、背景
传统卷积的权重是固定的，但不同的输入特征可能需要不同的卷积核权重。通过引入动态生成的注意力权重，可
以让卷积操作更具灵活性，适应输入特征的变化。
二、实现方法
使用一个轻量的卷积模块生成动态注意力权重，并将其与标准卷积的输出相结合，形成注意力卷积。
"""

class AttentionConvolution(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction=16):
        """
        注意力卷积模块
        :param in_channels: 输入通道数，输出通道数与输入通道数一致
        :param kernel_size: 卷积核大小
        :param reduction: 通道压缩比例
        """
        super(AttentionConvolution, self).__init__()

        # 卷积保持通道数不变
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),  # 恢复维度
            nn.Sigmoid()  # 激活
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征图
        :return: 注意力卷积后的输出特征图
        """
        # 标准卷积输出
        conv_out = self.conv(x)
        # 生成注意力权重
        attention_weights = self.attention(x)
        # 加权卷积输出
        output = conv_out * attention_weights
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
    attention_conv = AttentionConvolution(in_channels=64, kernel_size=3)
    output_tensor = attention_conv(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
