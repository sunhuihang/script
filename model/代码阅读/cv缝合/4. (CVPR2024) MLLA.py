import torch
import torch.nn as nn
from thop import profile, clever_format
import torch.nn.functional as F
import math

"""
MLLA：用于高分辨率视觉任务的类曼巴线性注意力模型 CVPR 2024

MLLA 模块背景
MLLA（Mamba-Like Linear Attention）模块基于曼巴（Mamba）模型的成功经验提出，旨在优化高分辨率视觉任务中的注意力机制。
传统的线性注意力模型具有计算效率高的优势，但在实际应用中表现相对不足。曼巴模型通过遗忘门和改进的模块设计在高分辨率输入处理中表现出色，
启发了 MLLA 模型的设计。与曼巴模型类似，MLLA 模型通过整合局部和全局的注意力机制，展现出卓越的性能。

MLLA 的局限性与创新
MLLA 模型旨在克服传统线性注意力模型在视觉任务中表现不佳的局限性。通过引入曼巴模型的核心创新设计，MLLA 模型解决了以下问题：
1. 遗忘门的替代：遗忘门在曼巴模型中用以增强顺序敏感性和局部偏差，但会引入递归计算，降低并行性。MLLA 模型使用位置编码（如RoPE、LePE）
替代遗忘门，既能提供必要的顺序信息，也能实现并行计算，加速推理速度。
2. 改进的块设计：MLLA 模块采用了类曼巴的块设计，将线性注意力与局部位置编码（LePE）和旋转位置编码（RoPE）相结合，
提升了在高分辨率图像任务中的特征建模能力。

MLLA 的创新点
MLLA 在以下几个方面做出了重要改进：
1. 局部与全局注意力结合：通过LePE和RoPE位置编码的组合，MLLA 能够灵活处理不同尺度的特征，从而提升在图像分类和密集预测任务中的性能。
2. 高并行性：去除了递归计算的限制，采用位置编码代替遗忘门，支持并行运算，从而实现更高的推理速度。
3. 多任务适用性：MLLA 支持在图像分类、目标检测和语义分割等任务中广泛应用，实验表明其在这些任务中的性能均优于多种视觉曼巴模型。

MLLA 的适用任务
作为一种即插即用的注意力模块，MLLA 适用于图像分类、目标检测和语义分割等多种计算机视觉任务。在 ImageNet、COCO 和 ADE20K 
等多个数据集上的实验结果表明，MLLA 模型不仅在精度上超越了曼巴模型，而且在推理速度上更具优势。MLLA 还可方便地集成到 Swin Transformer 
等结构中，显著提升了在非自回归视觉模型中的性能，其在高分辨率视觉任务中表现出色。
"""
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """

    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def generate_rotations(self, x):
        # 获取输入张量的形状
        *channel_dims, feature_dim = x.shape[1:-1][0], x.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0, "Feature dimension must be divisible by 2 * k_max"
        # 生成角度
        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=x.dtype, device=x.device) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d, dtype=x.dtype, device=x.device) for d in channel_dims],
                                           indexing='ij')], dim=-1)
        # 计算旋转矩阵的实部和虚部
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        return rotations

    def forward(self, x):
        # 生成旋转矩阵
        rotations = self.generate_rotations(x)
        # 将 x 转换为复数形式
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        # 应用旋转矩阵
        pe_x = torch.view_as_complex(rotations) * x_complex
        # 将结果转换回实数形式并展平最后两个维度
        return torch.view_as_real(pe_x).flatten(-2)


class MLLAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim=3, input_resolution=[160, 160], num_heads=4, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        x = x.reshape((x.size(0), x.size(2) * x.size(3), x.size(1)))
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        # self.rope = RoPE(shape=(h, w, self.dim))
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x.transpose(2, 1).reshape((b, c, h, w))
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 128, 128)
    image = torch.rand(*image_size)
    # Model
    model = MLLAttention(64) #MLLA
    out = model(image)
    print('input_size:', image.size())
    print('output_size:', out.size())
