import torch
import torch.nn as nn
"""
CV缝合救星魔改创新10：引入非对称卷积
一、背景
传统的卷积操作通常是对称的（如 3x3 或 5x5 卷积核），但对称卷积在捕获非对称特征时可能不够高效。通过引入非对称卷积（如 1x3 和 3x1），
可以更有效地捕获特征图中水平和垂直方向的信息。
二、实现方法
将SE模块中的普通卷积替换为非对称卷积，分解为多个方向性卷积操作（如 1x3 和 3x1），并融合这些特征以提升注意力机制的表达能力。
"""
class AsymmetricConvSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        使用非对称卷积的SE模块。
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(AsymmetricConvSEBlock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1_1x3 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv1_3x1 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # 使用全局平均池化提取通道信息
        avg_pool = self.global_avg_pool(x)
        y = self.conv1_1x3(avg_pool)  # 1x3 非对称卷积
        y = self.conv1_3x1(y)  # 3x1 非对称卷积
        y = self.relu(y)
        y = self.conv2(y)  # 恢复通道维度
        y = self.sigmoid(y).view(b, c, 1, 1)  # 生成注意力权重

        # 按通道加权输入特征
        output = x * y
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)
    asymmetric_conv_se_block = AsymmetricConvSEBlock(in_channels=64, reduction=16)
    output_tensor = asymmetric_conv_se_block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
