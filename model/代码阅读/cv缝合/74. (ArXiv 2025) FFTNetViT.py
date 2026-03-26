import torch
import torch.nn as nn
import torch.nn.functional as F

# 哔哩哔哩：CV缝合救星
"""
74. The FFT Strikes Again: An Efficient Alternative to Self-Attention (Arxiv 2025)
    FFT 再度出击：自注意力机制的高效替代方案
    即插即用模块：FFTNetViT（替身模块）
一、背景
传统自注意力机制在处理长序列时存在二次复杂度问题，计算和内存成本高，这对涉及长序列的任务，如长文
本语言建模和大规模图像处理十分不利。为解决该问题，本文提出 FFTNetViT 模块，借助快速傅里叶变换
（FFT）实现高效的全局令牌混合，降低计算复杂度。

二、FFTNetViT 原理
1. 整体架构设计
FFTNetViT 将输入序列映射到频域，利用 FFT 捕捉长距离依赖关系，通过自适应频谱滤波和非线性激活等操
作，再经逆 FFT 变换回令牌域，实现全局混合表示。同时，它还可选择融合局部窗口（STFT 或小波变换）和
全局 FFT 分支，综合捕捉长距离和短距离交互信息。
2. 核心组件及操作
A. 全局傅里叶变换：对整个输入序列应用 FFT，得到频域表示，在不进行显式成对比较的情况下捕获长距离
依赖。
B. 局部窗口化（可选）：通过 STFT 或小波变换处理局部上下文。STFT 将输入划分为重叠窗口，应用窗口函
数后进行 FFT；小波变换则分解局部段获取多尺度时变频率信息，保留高分辨率局部结构。
C. 融合全局和局部表示：在频域中，通过等距融合（或门控）方式合并全局和局部表示。等距融合是将两者特
征按维度拼接，再用可学习的等距矩阵融合；门控则根据全局上下文学习标量或向量门，调制全局和局部变换的
融合。
D. 自适应频谱滤波：计算全局上下文向量，输入小 MLP 生成每个频率 bin 和注意力头的缩放与偏置参数，对
融合后的频率信息进行加权和偏置调整。
E. 非线性激活和逆 FFT：应用 modReLU 激活函数增强表示能力，最后通过逆 FFT 将处理后的频域信息转换回
令牌域，得到最终输出。

三. 微观设计考量
1. 高效混合：FFT 以 O (n log n) 的时间复杂度实现全局混合，结合局部窗口化操作，能在近似 O (n log n)
时间内融合全局和局部信息。自适应滤波器依全局上下文对频率成分隐式掩码，modReLU 捕捉高阶结构，提升模型
表达能力。
2. 理论保障：Parseval 定理确保信号在傅里叶变换下能量守恒，DFT 的正交分解和酉性质保证变换过程中内积结构
不变，使模型能有效保留输入信息。
3. 计算复杂度优势：整体计算复杂度为 O (d n log n)，实际应用中接近 O (n log n)。相比传统自注意力机制的 
O (n²) 复杂度，FFTNetViT 计算成本显著降低，在处理长序列时更具优势。

四、适用任务
1. 长距离竞技场（LRA）任务
涵盖 ListOps、Text、Retrieval、Image、Pathfinder 和 Path-X 等任务。FFTNetViT 在多数任务上优于传统 
Transformer 和 FNet，添加局部窗口化（STFT）后平均准确率进一步提升，展现出良好的性能。
2. 图像分类任务
在 ImageNet 分类任务中，FFTNetViT 相比标准 ViT，使用更少的 FLOPs。添加局部窗口化（STFT）后，Top-1 和 
Top-5 准确率有小幅度但稳定的提升，超过 ViT 基线。同时，FFTNetViT 在处理图像时的吞吐量更高，速度更快，且
随着批量大小增加，优势更明显。
3. 消融实验体现优势的任务
通过在 ImageNet（Base 变体，无窗口化）上的消融实验表明，FFTNetViT 的频谱门控、自适应模块和 FFT-based 
滤波等组件均对性能有重要贡献。其中，用卷积替换 FFT 层会导致准确率大幅下降，突出了 FFT 在模型中的关键作用。
"""

class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)
    def forward(self, x):
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))
class FFTNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.filter = nn.Linear(dim, dim)
        self.modrelu = ModReLU(dim)
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        x_fft = torch.fft.fft(x, dim=1)  # FFT along the sequence dimension
        x_filtered = self.filter(x_fft.real) + 1j * self.filter(x_fft.imag)
        x_filtered = self.modrelu(x_filtered)
        x_out = torch.fft.ifft(x_filtered, dim=1).real
        return x_out
if __name__ == '__main__':
    # 参数设置
    batch_size = 1      # 批量大小
    seq_len = 224 * 224 # 序列长度(Transformer 中的 token 数量)
    dim = 32      # 维度
    # 创建随机输入张量,形状为 (batch_size, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, dim)
    # 初始化 FFTNetBlock 模块
    model = FFTNetBlock(dim = dim)
    print(model)
    print("哔哩哔哩: CV缝合救星!")
    output = model(x)
    print(x.shape)
    print(output.shape)