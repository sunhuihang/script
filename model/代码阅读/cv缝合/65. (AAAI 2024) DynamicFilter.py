import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple

# B站：CV缝合救星
""" 
65. FFT-based Dynamic Token Mixer for Vision （AAAI 2024 顶会论文）
即插即用模块：Dynamic Filter（MHSA 平替）

一、背景：
Transformer 架构在计算机视觉领域应用广泛，其中的多头自注意力（MHSA）机制虽然性能不错，但计算复杂度高，
处理高分辨率图像时速度慢。基于 FFT 的全局滤波器虽有计算优势，还能捕捉低频信息，却没有和最新架构很好地结合。
所以，本文提出 Dynamic Filter 模块，来解决这些问题，提升模型在高分辨率图像任务中的表现。

二、Dynamic Filter 原理
1. 整体架构设计：Dynamic Filter 模块用来替代 MHSA，它基于 MetaFormer 架构，能为图像的每对特征通道动态
生成全局滤波器，通过动态确定合适的全局滤波器，有效处理特征，进而提升模型性能。
2. 核心组件
A. 动态滤波器生成器：由神经网络来动态确定合适的全局滤波器。它有 N 维的全局滤波器基，通过 MLP（M）确定线性
组合系数，为每个通道生成动态滤波器。
B. MLP 加权：通过特定的计算方式调整滤波器系数的权重，具体涉及到层归一化、激活函数以及一些矩阵运算，通过调整
中间维度与输入维度的比例，实现对滤波器系数的加权。

三.、微观设计考量：
Dynamic Filter 模块优势显著。它可以动态生成滤波器，比传统全局滤波器更灵活；基于 MetaFormer 架构，缩小了和
顶尖模型在精度上的差距；在处理高分辨率图像的下游任务（像密集预测）时，计算成本更低，在高分辨率下，吞吐量更高，
内存需求更低。

四、适用任务
1. 图像分类：在 ImageNet-1K 数据集上，基于 Dynamic Filter 的 DFFormer 和 CDFFormer 模型，与多种模型比较
，在不使用注意力或保留机制的模型里，top-1 准确率表现很突出。DFFormer 在传统基于 FFT 的模型中性能领先，比其他
同类模型效果好很多，还超过了和 FFT 常对比的 MLP - 基于模型。CDFFormer 结合了卷积，性价比更高，最大的 
CDFFormer-B36 模型比 DFFormer-B36 表现更好。
2. 语义分割：在 ADE20K 数据集上，以 Semantic FPN 为框架训练测试，基于 DFFormer 和 CDFFormer 的模型在语义
分割任务上很有效，比基于其他模型（比如 PoolFormer）的方法表现更好。例如，DFFormer-S36 的 mIoU 比 PoolFormer-S36 
高 5.5，CDFFormer-M36 达到 48.6 mIoU。
3. 目标检测：在 COCO 基准上，以 RetinaNet 为框架，用在 ImageNet-1K 数据集上预训练的 DFFormer 和 CDFFormer 作为
骨干网络，比 ResNet 和 PoolFormer 等骨干网络表现更出色。比如，DFFormer-S36 的 AP 比 PoolFormer-S36 高 5.8，证明
了其在目标检测任务中的有效性。
"""
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x

if __name__ == '__main__':
    block = DynamicFilter(32, size=64)  # size==H,W

    # 若input形状为B C H W，先用下面代码变换张量形状
    input = torch.rand(3, 32, 64, 64)   # 输入 B C H W
    input_bhwc = input.permute(0, 2, 3, 1)  # B H W C

    output = block(input_bhwc)

    output = output.permute(0, 3, 1, 2)  # B C H W
    print(input.size())
    print(output.size())  # 输出的形状
