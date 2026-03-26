import torch
import torch.nn as nn
"""
63. Focal Modulation Networks （NeurIPS 2022顶会）
即插即用模块：Focal Modulation（VIT平替）

一、背景
Transformer 中的自注意力机制（SA）在计算机视觉领域取得了显著成果，但因其对视觉令牌数量
具有二次复杂度，在处理高分辨率输入时效率受限。为解决这一问题，许多研究提出了 SA 的变体或
结合卷积的模型。在此背景下，本文提出 Focal Modulation 模块，旨在寻找一种比 SA 更好的
对输入依赖的长距离交互进行建模的方法。

二、Focal Modulation 原理
1. 整体架构设计：Focal Modulation 模块用于取代 SA 在视觉模型中对令牌交互进行建模。它
通过早期聚合过程生成精炼表示，与 SA 的晚期聚合过程不同。具体而言，该模块先对每个查询位置
周围的上下文进行聚焦聚合，再用聚合后的上下文自适应地调制查询，从而实现输入依赖的令牌交互，
同时简化了交互过程，使交互仅基于少量特征，降低计算复杂度。
2. 核心组件：
A. 聚焦上下文编码：利用堆叠的深度卷积层实现，从短距离到长距离编码视觉上下文。通过一系列
深度卷积操作，在不同粒度级别上提取上下文特征，为后续的聚合和调制提供丰富的信息。
B. 门控聚合：根据查询内容，选择性地将不同粒度级别的上下文特征聚合到一个调制器中。通过计
算空间和级别感知的门控权重，对不同级别的特征图进行加权求和，实现自适应的上下文聚合，使模
型能够根据查询的需求动态地获取合适的上下文信息。
C. 逐元素仿射变换：将调制器注入到查询中，通过逐元素乘法操作，实现对查询的调制，从而生成
最终的输出特征。这种操作方式能够在通道和空间上对查询进行特异性调制，增强模型对不同特征的
表达能力。
3. 微观设计考量：Focal Modulation 模块具有多项优势特性。
它具有平移不变性，由于查询投影函数和上下文聚合函数始终以查询令牌为中心，且未使用位置嵌入，
因此对输入特征图的平移具有不变性；具有显式的输入依赖性，调制器通过聚合目标位置周围的局部特
征计算得出，明确依赖输入内容；具备空间和通道特异性，目标位置作为上下文聚合函数的指针实现空
间特异性调制，逐元素乘法实现通道特异性调制；实现了解耦的特征粒度，查询投影函数保留单个令牌
的精细信息，上下文聚合函数提取更粗糙的上下文信息，二者通过调制相互结合，有效整合不同粒度的
信息，增强特征表达能力。
三、适用任务
1. 图像分类：在 ImageNet-1K 分类任务中，FocalNet-T、FocalNet-S 和 FocalNet-B 模型
取得了优异成绩。例如，FocalNet-T（LRF）达到了 82.3% 的 Top-1 准确率，超过了 Swin-Tiny 
等模型。通过模型增强实验发现，常用的技术如重叠补丁嵌入和加深变薄网络结构等可进一步提升 
FocalNets 的性能。在 ImageNet-22K 预训练后，FocalNets 在不同分辨率下的微调表现也优于
Swin 等模型，证明了 Focal Modulation 模块在图像分类任务中的有效性。
2. 目标检测和实例分割：在 COCO 2017 目标检测任务中，以 Mask R-CNN 为检测方法，FocalNet-T/S/B
作为骨干网络，相比 Swin Transformer 和 FocalAtt 等模型，FocalNets 在不同训练调度下均显著提升
了检测精度，如 FocalNet-T（SRF）在 1× 调度下 box mAP 相比 Swin-Tiny 提升了 2.2。在实例分割
任务中也呈现出类似的优势趋势。此外，使用不同检测方法（Cascade Mask R-CNN、Sparse RCNN 和 ATSS）
与 FocalNet-T 结合训练，均取得了优于之前方法的成绩，验证了 FocalNets 在目标检测领域的通用性和有
效性。
3. 语义分割：在 ADE20K 语义分割任务中，使用 UperNet 作为分割方法，FocalNet-T/S/B 作为骨干网络，
FocalNets 在单尺度和多尺度评估下均显著优于 Swin 和 Focal Transformer 等模型。例如，FocalNet-B（SRF）
在单尺度评估下 mIoU 比 Swin Transformer 高出 2.1。进一步扩大模型规模，使用 FocalNet-L 在 ADE20K 和 
COCO 全景分割任务中，也取得了优异的成绩，超过了 Swin-L 等模型，证明了 Focal Modulation 模块在语义分割
任务中的有效性，尤其在处理高分辨率密集预测任务时表现出色。
"""


class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        flops += N * self.dim * (self.dim * 2 + (self.focal_level + 1))

        # focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k] ** 2 + 1) * self.dim

        # global gating
        flops += N * 1 * self.dim

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


# 输入 B H W C,  输出 B H W C
if __name__ == '__main__':
    block = FocalModulation(dim=64, focal_window=3, focal_level=2)
    input = torch.rand(3, 56, 56, 64)
    output = block(input)
    print(input.size())
    print(output.size())