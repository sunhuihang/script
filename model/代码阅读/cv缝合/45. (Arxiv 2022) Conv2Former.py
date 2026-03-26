import torch
from torch import nn
import torch.nn.functional as F
"""
Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition
Pattern Analysis and Machine Intelligence (2024 SCI 1区TOP IF23.6) 
即插即用模块：ConvMod（替身模块）
一、背景
卷积神经网络（ConvNets）在视觉识别领域一直占据重要地位，但早期模型如 VGGNet、Inception系列和ResNet系列等，
在全局上下文信息建模方面存在不足。尽管 SENet 系列引入了注意力机制提升性能，Vision Transformers（ViTs）的
出现仍推动了视觉识别模型的进一步发展，其自注意力机制能更好地编码空间信息，但计算成本高。ConvNeXt 通过调整 
ResNet 的训练方法和空间卷积位置取得良好效果，RepLKNet 等探索了大核卷积潜力，如何高效利用卷积构建强大 
ConvNet 架构仍是研究热点。Conv2Former 旨在探索更高效利用卷积编码空间特征的方法，通过对比 ConvNets 和 ViTs
的设计原则，提出卷积调制操作简化自注意力，构建了性能出色的 Conv2Former 网络家族，在多种视觉任务中表现优异。
二、ConvBlock 原理
1. 整体架构设计
A. 金字塔架构与阶段特征：Conv2Former 采用金字塔架构，类似 ConvNeXt 和 Swin Transformer，包含四个阶段，
各阶段特征图分辨率不同。相邻阶段用 Patch Embedding 块降分辨率，多为 2×2 卷积且步长 2。不同阶段卷积块数量
不同，如 Conv2Former-T 在四个阶段的卷积块数量分别为 3、3、12、3。由此形成层次化结构，适应不同层次特征提取
需求。
B. 模型变体与参数设置：有 Conv2Former-N、Conv2Former-T 等五种变体，参数数量和计算复杂度各异。如 Conv2
Former-N 参数 15M，FLOPs 2.2G；Conv2Former-B 参数 90M，FLOPs 15.9G。通过不同配置满足不同规模视觉识别
任务，为模型选择提供多种可能。
2. 卷积调制块核心组件
A. 自注意力机制原理与局限：传统自注意力机制对输入令牌序列，先经线性层生成键、查询和值，输出是值基于相似度得分
的加权平均，相似度得分矩阵计算复杂，随序列长度增加计算成本呈二次增长，处理高分辨率图像时计算量大。
B. 卷积调制操作流程：Conv2Former 的卷积调制层，用深度卷积和 Hadamard 乘积简化自注意力。输入令牌经深度卷积
计算权重，与线性投影后的值进行 Hadamard 乘积得输出。此操作使空间位置与周围像素关联，通过线性层实现通道信息交
互，输出为区域像素加权和。
C. 优势对比：与自注意力相比，Conv2Former 用卷积建立关系，处理高分辨率图像内存效率更高；与经典残差块相比，其
调制操作能更好适应输入内容，在特征编码上独具优势。
3. 微观设计考量
A. 大核卷积利用策略：传统 ConvNets 多以 3×3 卷积构建，ConvNeXt 扩展到 7×7 有性能提升，但更大核通常无增益且增
计算负担。Conv2Former 不同，核大小从 5×5 到 21×21 性能持续提升，如 Conv2Former-T 和 Conv2Former-B。
考虑效率，默认核大小设为 11×11，有效提升大核卷积利用效率。
B. 加权策略特点：Conv2Former 将深度卷积输出作权重调制特征，Hadamard 乘积前不使用激活或归一化层是关键。实验表明，
添加 Sigmoid 函数会降低性能。该加权策略与其他类似方法动机不同，更注重简化自注意力和利用大核卷积空间信息。
C. 归一化与激活层选择：遵循 ViT 和 ConvNeXt，采用 Layer Normalization 和 GELU 激活层，这种组合带来一定性能
增益，有助于模型稳定训练和高效性能表现。

三、适用任务
Conv2Former 适用于众多视觉识别任务，如 ImageNet 分类、COCO 对象检测 / 实例分割、ADE20k 语义分割等。在这些任务中，
它的性能优于许多现有流行网络架构，为视觉识别领域提供了强大且通用的解决方案，可广泛应用于各类视觉识别场景。
"""

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        x = a * v
        x = self.proj(x)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = ConvMod(64).cuda()
    input = torch.rand(3, 64, 85, 85).cuda()
    output = block(input)
    print(input.size(), output.size())
