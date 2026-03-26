import torch
import torch.nn as nn
"""
CV缝合救星魔改创新9：改用最大值注意力
一、背景
传统的SE模块使用全局平均池化（Global Average Pooling, GAP）来提取通道的全局信息，但这种方式容易忽略特征图中显著区域的信息。
通过引入全局最大池化（Global Max Pooling, GMP），我们可以专注于提取特征图中最显著的激活信息，从而提升注意力机制对关键特征的捕
获能力。
二、实现方法
1. 将全局平均池化替换为全局最大池化
2. 结合平均池化和最大池化的结果来生成更加丰富的通道注意力权重。
"""
class MaxAttentionSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        改用最大值注意力的SE模块。
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(MaxAttentionSEBlock, self).__init__()

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # 使用全局最大池化提取通道信息
        max_pool = self.global_max_pool(x).view(b, c)
        y = self.fc1(max_pool)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)  # 生成注意力权重

        # 按通道加权输入特征
        output = x * y
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)
    max_attention_se_block = MaxAttentionSEBlock(in_channels=64, reduction=16)
    output_tensor = max_attention_se_block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
