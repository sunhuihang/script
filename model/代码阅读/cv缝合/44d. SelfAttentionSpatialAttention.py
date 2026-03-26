import torch
import torch.nn as nn

"""
自注意力机制空间注意力 (Self-Attention Based Spatial Attention)
一、背景
传统的空间注意力无法捕捉特征图中不同位置之间的长距离依赖，而自注意力机制通过点积操作可以有效建模这些关系。
二、实现方法
通过计算特征图中任意两个位置之间的关系权重，生成一个空间注意力图，并使用该注意力图调整特征图的值。
"""

class SelfAttentionSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        """
        自注意力机制空间注意力模块
        :param in_channels: 输入通道数
        """
        super(SelfAttentionSpatialAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征图
        :return: 自注意力加权后的输出特征图
        """
        batch_size, channels, height, width = x.size()

        # 生成 Query, Key, Value
        query = self.query_conv(x).view(batch_size, -1, height * width)  # (B, C/8, H*W)
        key = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C/8, H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, H*W)

        # 计算自注意力关系权重
        attention = torch.bmm(query.permute(0, 2, 1), key)  # (B, H*W, H*W)
        attention = self.softmax(attention)  # 对最后一个维度做归一化

        # 根据注意力权重加权 Value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, channels, height, width)  # 恢复为特征图形状

        # 将注意力结果加到输入特征图上（残差连接）
        output = x + out
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
    self_attention = SelfAttentionSpatialAttention(in_channels=64)
    output_tensor = self_attention(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
