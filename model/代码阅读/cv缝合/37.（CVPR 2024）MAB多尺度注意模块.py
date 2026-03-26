import torch
import torch.nn as nn
import torch.nn.functional as F
'''
MAB多尺度注意力模块（CVPR 2024）
即插即用模块：MAB（替身增强模块）
一、背景
在单图像超分辨率（SISR）任务中，从低质量输入重建高频信息存在困难，CNN和基于Transformer的模型通过不同方式提高重建质量。
CNN通过扩大模型容量（如使用更大数据集和更好策略）或激活更多图像内信息（如扩大感知域、采用复杂拓扑和注意力机制）来提升
性能，但面临训练负担和数据收集消耗增加、过度训练和高成本等问题；Transformer 模型在自注意力方面表现出色，但存在计算复
杂等问题。为了在超分辨率任务中充分发挥ConvNet的潜力，本文提出了MAB模块。

二、MAB 模块原理
1. 网络架构：MAN网络由浅特征提取模块（SF）、基于多个多尺度注意力块（MAB）的深特征提取模块（DF）和高质量图像重建模块组成。
输入LR图像经SF提取原始特征，再由DF中的MAB进一步提取，最后通过重建模块恢复 HQ 图像。优化时采用损失。
2. 多尺度注意力块（MAB）：由多尺度大核注意力（MLKA）模块和门控空间注意力单元（GSAU）组成。输入特征先进行层归一化，然后
经过 MLKA 和 GSAU 处理，并通过元素级乘法和卷积操作，最后与输入相加得到输出。采用层归一化可保留实例细节并加速收敛。
3. 多尺度大核注意力（MLKA）：由大核注意力（LKA）、多尺度机制和门控聚合组成。LKA 通过分解大核卷积建立长距离关系；多尺度机
制将输入特征分组，采用不同参数的 LKA 生成多尺度注意力图；门控聚合利用空间门动态调整 LKA 输出，避免块效应并学习更多局部信
息。通过控制计算成本，灵活捕捉局部和全局信息。
4. 门控空间注意力单元（GSAU）：将简单空间注意力（SSA）和门控线性单元（GLU）集成到前馈网络中，通过深度卷积加权特征图，
应用空间门去除非线性层，在降低参数和计算量的同时捕获局部连续性。
5. 大核注意力尾（LKAT）：在深度提取骨干的尾部引入7 - 9 - 1 LKA，通过两个 1×1 卷积包裹，以总结更合理的信息，提高最重建特征的代表性。
三、适用任务
1. 图像超分辨率任务：适用于不同缩放因子（如 ×2、×3、×4）的单图像超分辨率任务，能在经典和轻量级模型中实现性能与复杂度的权衡，
有效提高重建图像的质量，恢复更多细节。
2. 其他低层次计算机视觉任务：可作为一种有效的特征提取和增强模块，为其他低层次视觉任务提供参考，帮助提升模型在处理相关任务时对
特征的表示能力和信息聚合能力。
'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class GSAU(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor=15, attn='GLKA'):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats
        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)], dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x
 # MAB
class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)
    def forward(self, x):
        # large kernel attention
        x = self.LKA(x)
        # local feature extraction
        x = self.LFE(x)
        return x

if __name__ == "__main__":
    input = torch.randn(1, 30, 128, 128)
    MAB = MAB(30)
    output = MAB(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
