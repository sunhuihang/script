import torch
import torch.nn as nn
"""
CV缝合救星魔改创新7：融合自注意力机制
一、背景
传统SE模块只在通道维度生成注意力权重，缺乏捕获全局特征的能力。自注意力机制（Self-Attention）可以通过计算特征图中所有位置
之间的相关性来捕获全局上下文信息，从而提升特征表达能力。
二、实现方法
将自注意力机制与SE模块结合，计算特征图中每个位置的全局相关性，并结合通道注意力生成更加全面的注意力权重。
"""
class SelfAttentionSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        融合自注意力机制的SE模块。
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(SelfAttentionSEBlock, self).__init__()

        # SE模块
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 自注意力机制
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. 通道注意力机制
        avg_pool = self.global_avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc1(avg_pool)
        y = nn.ReLU()(y)
        y = self.fc2(y)
        y_channel = self.sigmoid(y).view(b, c, 1, 1)  # 通道权重

        # 2. 自注意力机制
        q = self.query(x).view(b, -1, h * w)  # [B, C//8, H*W]
        k = self.key(x).view(b, -1, h * w)  # [B, C//8, H*W]
        v = self.value(x).view(b, c, h * w)  # [B, C, H*W]

        attention = torch.bmm(q.permute(0, 2, 1), k)  # [B, H*W, H*W]
        attention = self.softmax(attention)  # 归一化
        attention = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, H*W]
        attention = attention.view(b, c, h, w)  # 恢复形状

        # 3. 融合通道和自注意力
        output = x * y_channel + attention  # 通道和自注意力加权融合
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)
    self_attention_se_block = SelfAttentionSEBlock(in_channels=64, reduction=16)
    output_tensor = self_attention_se_block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
