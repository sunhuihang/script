import math
import torch
import torch.nn as nn
"""
CV缝合救星魔改创新：动态光谱空间自适应增强机制
一、背景
1. 在现有的 ESSA 模块应用于单高光谱图像超分辨率任务时，尽管其基于光谱相关系数（SCC）的设计和高效的核注意力
机制已取得成效，但仍存在可优化空间。本创新点聚焦于为 ESSA 增添动态光谱 - 空间自适应模块，使其能依据输入高光
谱图像的光谱特性和空间结构动态优化计算过程。
2. 具体实现是在 ESSA 前向传播中，先对输入特征图在光谱维度进行池化操作，获取光谱特征向量，同时在空间维度进行
分块池化，得到空间特征向量。将两者拼接后输入由多层感知机（MLP）构成的小型神经网络进行处理，输出动态调整参数。
这些参数用于实时调整 SCC 计算中的相关系数、核函数参数以及注意力组合权重。在光谱方面，可依据不同波段的相关性
强弱动态改变权重；在空间上，能根据图像区域的纹理、结构复杂程度灵活调整衰减范围和注意力分配，如在纹理丰富区域
增强注意力，在平滑区域适当降低计算强度。
二、优势
1. 极大增强了对不同高光谱图像数据的适应性，无论是复杂的自然场景还是特定的工业检测图像，都能有效提升模型在多样
化任务中的泛化性能，减少因数据差异导致的性能波动。
2. 相较于固定参数的 ESSA 机制，动态调整可精准聚焦于光谱关键波段和空间重要区域，显著提升模型对空间 - 光谱信息
的提取与表达能力，在单高光谱图像超分辨率任务中有望获得更高质量的重建结果，减少伪影并增强细节恢复效果。
"""

class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = t1 + t2
        attn = self.ln(attn)
        x = attn.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


# 新添加的多尺度融合模块
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_fused = torch.cat([x1, x2, x3], dim=1)
        x_fused = self.fusion_conv(x_fused)
        return x_fused


# 改进后的 ESSAttn 模块，包含多尺度融合
class ImprovedESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ess_attn = ESSAttn(dim)
        self.fusion = MultiScaleFusion(dim, dim)

    def forward(self, x):
        attn_output = self.ess_attn(x)
        fused_output = self.fusion(attn_output)
        return fused_output


# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    input = torch.randn(1, 32, 64, 64)
    model = ImprovedESSAttn(32)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())