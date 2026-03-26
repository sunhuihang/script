
"""
CV缝合救星魔改创新2：动态通道选择机制
描述：在SHViT中，将部分通道（即pdim通道）用于单头注意力，而剩余通道则保持不变。可以增加一个动态通道选择模块，
使模型能够自适应选择哪些通道参与注意力计算。这种动态选择机制可以根据输入特征图的内容决定使用哪些通道，从而更
灵活地分配计算资源。

实现思路：
增加一个轻量的通道注意力模块（如SE模块）或基于门控机制的模块，让模型根据输入特征自动选择部分通道进行注意力计算。
使用一个小的MLP或门控层，输入每个通道的全局池化结果（例如平均池化后的通道信息），输出一个权重矩阵，该权重矩阵
可以选择性地放大或减小通道的重要性。

只对被选择的通道进行单头自注意力操作，未选择的通道保持原样。这样在保持计算效率的同时可以增加模型的特征多样性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicChannelSelect(nn.Module):
    """
    动态通道选择模块：基于通道注意力选择部分通道进行注意力计算。
    """

    def __init__(self, in_channels, select_channels, reduction_ratio=4):
        super(DynamicChannelSelect, self).__init__()
        self.select_channels = select_channels

        # 生成通道权重的轻量模块
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算通道注意力权重
        scale = self.avg_pool(x)
        scale = self.fc(scale)

        # 对输入张量加权
        x = x * scale
        # 分成参与注意力的通道和未改变的通道
        attn_part, non_attn_part = torch.split(x, [self.select_channels, x.size(1) - self.select_channels], dim=1)
        return attn_part, non_attn_part


class SHSA(nn.Module):
    """
    单头自注意力模块：仅对选择的通道执行自注意力。
    """

    def __init__(self, dim, attn_dim, select_channels):
        super(SHSA, self).__init__()
        self.scale = attn_dim ** -0.5
        self.attn_dim = attn_dim
        self.select_channels = select_channels

        # 计算查询、键和值
        self.qkv = nn.Conv2d(select_channels, attn_dim * 2 + select_channels, kernel_size=1)
        self.proj = nn.Conv2d(select_channels, select_channels, kernel_size=1)

    def forward(self, x):
        # 分割查询、键和值
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.attn_dim, self.attn_dim, self.select_channels], dim=1)

        # 展平并计算注意力
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 应用注意力权重
        out = (v @ attn.transpose(-2, -1)).reshape(x.size(0), self.select_channels, x.size(2), x.size(3))
        return self.proj(out)


class SHViTBlockWithDynamicSelect(nn.Module):
    """
    带动态通道选择的 SHViT 块：结合动态通道选择和单头自注意力。
    """

    def __init__(self, dim, attn_dim=16, select_channels=32, reduction_ratio=4):
        super(SHViTBlockWithDynamicSelect, self).__init__()
        self.dynamic_select = DynamicChannelSelect(dim, select_channels, reduction_ratio)
        self.shsa = SHSA(dim, attn_dim, select_channels)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )

    def forward(self, x):
        # 通过动态通道选择模块
        attn_part, non_attn_part = self.dynamic_select(x)

        # 单头自注意力应用在选择的通道部分
        attn_out = self.shsa(attn_part)

        # 拼接注意力和未改变的通道部分
        out = torch.cat([attn_out, non_attn_part], dim=1)

        # 通过前馈网络
        return self.ffn(out)


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)  # 输入特征图
    shvit_block = SHViTBlockWithDynamicSelect(dim=64, attn_dim=16, select_channels=32)  # 初始化模块
    output_tensor = shvit_block(input_tensor)  # 前向传播
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")







