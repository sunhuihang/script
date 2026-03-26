import torch
from torch import nn

# 哔哩哔哩：CV缝合救星
"""
72. SLAB: Efficient Transformers with Simplified Linear Attention and Progressive 
    Re-parameterized Batch Normalization (ICML 2024 顶会论文)
    SLAB：具有简化线性注意力和渐进式重参数化批量归一化的高效变换器 (华为诺亚方舟实验室)
    即插即用模块：SLAB（替身模块）
    
一、背景
Transformer 在多领域表现出色，但计算成本高，其归一化层和注意力模块是效率瓶颈。LayerNorm 推理时
计算统计量影响速度，BatchNorm 虽能加速推理，却会导致 Transformer 训练崩溃和性能下降。为解决这些
问题，提出 RepBN 模块。

二、RepBN 模块介绍
（一）整体设计
RepBN 是重新参数化的 BatchNorm，目的是逐步替代 Transformer 中的 LayerNorm。训练时，借助超参数
γ 控制 LayerNorm 和 RepBN 的输出比例，γ 从 1 渐变为 0，让模型从以 LayerNorm 为主过渡到纯 RepBN
架构，减少推理计算负担。
（二）核心组件与操作
1. 重新参数化设计：RepBN 在 BatchNorm 基础上添加可学习参数 η 与输入 X 的乘积项。训练结束后，能重
新参数化，改变输出分布的控制参数，调整模型对数据分布的适应性。
2. 渐进替换策略：通过特定公式实现从 LayerNorm 到 RepBN 的渐进替换。采用线性衰减策略调整 γ，在训练初
期利用 LayerNorm 稳定训练，后期切换到高效的 BatchNorm 架构。

三、微观设计考量
从训练优化看，渐进替换策略平衡了训练稳定性和推理效率。RepBN 的设计让模型前期稳定训练，后期提升推理效率。
实际应用中，线性衰减策略简单有效，降低训练复杂度。

四、适用任务
1. 主要用于 Transformer 相关任务：在图像分类任务里，使用 RepBN 的模型性能提升显著，像 DeiT-S 模型使用
PRepBN 后，Top-1 准确率比对比方法高。在对象检测和实例分割任务中，基于 RepBN 的模型与原模型性能相当，但推
理延迟更低 。
2. 可扩展到语言建模任务：在语言建模任务中，应用 RepBN 的模型与使用 LayerNorm 的模型相比，困惑度相似，延迟
却降低。在大语言模型如 LLaMA-350M 上应用 PRepBN，吞吐量提升，平均准确率也有提高。
"""

class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x) + self.alpha * x
        x = x.transpose(1, 2)
        return x


# 可扩展到四维, 哔哩哔哩:CV缝合救星
class RepBN2d(nn.Module):
    def __init__(self, channels):
        super(RepBN2d, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)  # 使用BatchNorm2d

    def forward(self, x):
        # BatchNorm2d处理四维输入数据
        x = self.bn(x) + self.alpha * x  # BatchNorm + alpha * input
        return x

if __name__ == "__main__":
    # 模块参数
    batch_size = 1  # 批大小
    channels = 32  # 输入特征通道数
    height = 256  # 图像高度
    width = 256  # 图像宽度

    model = RepBN2d(channels=channels)
    print(model)
    print("哔哩哔哩:CV缝合救星, NB!")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = model(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)