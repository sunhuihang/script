"""
二、多层次特征融合
1. 缺陷：目前SCSA模块主要基于单一层级的空间和通道注意力信息

2. 解决方案：引入多层次特征融合，例如在不同网络层级上对特征图进行聚合，或者在不同分辨率的特征图之间实现交互。
通过融合低层次的细节信息和高层次的语义信息，增强模型对复杂结构和细节的捕捉能力。进一步提升图像分割、目标检测等
任务中细节处理和物体边界识别的效果。提高模块对多语义空间信息的利用效率，使得特征表达更加丰富。
"""
import typing as t
import torch
import torch.nn as nn
from einops import rearrange
class MultiLevelSCSA(nn.Module):
    def __init__(self, dim, head_num, levels=3, group_kernel_sizes=[3, 5, 7, 9], qkv_bias=False, attn_drop_ratio=0.):
        super(MultiLevelSCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.levels = levels  # 表示要融合的特征层级数
        self.group_chans = dim // 4

        # 每层级独立的卷积和自注意力模块
        self.multi_layer_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
            for k in group_kernel_sizes
        ])
        self.multi_layer_norms = nn.ModuleList([nn.GroupNorm(4, dim) for _ in range(self.levels)])
        self.sa_gate = nn.Sigmoid()

        # 融合后的通道注意力
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)  # v矩阵
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def forward(self, x):
        layer_outputs = []
        for conv, norm in zip(self.multi_layer_convs, self.multi_layer_norms):
            layer_out = self.sa_gate(norm(conv(x)))
            layer_outputs.append(layer_out)

        # 多层级融合特征
        fused_features = sum(layer_outputs) / self.levels

        # 通道注意力
        q = self.q(fused_features)  # 查询（Query）
        k = self.k(fused_features)  # 键（Key）
        v = self.v(fused_features)  # 值（Value）

        # 计算注意力权重
        attn_weights = q @ k.transpose(-2, -1) * self.head_dim ** -0.5
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 用v加权特征并输出
        output = attn_weights @ v  # 加权后的输出

        return fused_features * output  # 返回加权后的融合特征


if __name__ == '__main__':
    input = torch.randn(1, 128, 64, 64)
    model = MultiLevelSCSA(dim=128,  head_num=8)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())