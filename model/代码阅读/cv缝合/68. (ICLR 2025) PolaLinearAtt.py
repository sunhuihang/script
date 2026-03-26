import torch
import torch.nn as nn

# B站:CV缝合救星
"""
68. PolarFormer: Polarity-aware Linear Attention for Vision Transformers (ICLR 2025).
    用于视觉Transformer的极性感知线性注意 哈尔滨工业大学，深圳，彭城实验室，澳大利亚昆士兰大学
    即插即用模块: PolarFormer (替身强化模块)
一、背景
Transformer 在视觉任务中取得成功，但自注意力机制的二次复杂度限制了其在长序列或高分辨率图像处理时的效率。
线性注意力虽降低了复杂度，但存在信息丢失问题，导致注意力图判别力下降、均匀性增加，无法有效区分强弱查询 - 
键对，性能表现不如基于 softmax 的注意力机制。为解决这些问题，提出 PolarFormer，旨在提升线性注意力的性
能，平衡表达能力和效率。

二、PolarFormer 模块介绍
（一）整体设计
PolarFormer 模块的设计目的B 站 Transformer 性能救星是解决线性注意力机制中因忽略负向交互和注意力图尖峰
特性丢失的问题。通过极性感知注意力机制，分离查询 - 键对的极性进行处理，并引入可学习的幂函数重新缩放，恢复
注意力图的尖峰特性，提升模型对查询 - 键交互的捕捉能力，增强模型在视觉任务中的表达能力和效率。
（二）核心组件与操作
1. 极性感知注意力：将查询向量和键向量分解为正负分量，分别计算相同符号和相反符号分量之间的交互，恢复被传统线
性注意力丢弃的负向信息。采用可学习的极性混合机制，避免直接相减带来的不稳定问题，通过对值向量按维度拆分并使用
可学习系数矩阵加权，更准确地重建原始 softmax 注意力权重。
2. 降低线性注意力熵：提出正序列熵（PSE）度量，并证明存在一类一阶和二阶导数均为正的逐元素函数可降低注意力分布
的熵。为简化模型，选用带可学习指数的幂函数，不同维度的指数可学习，以适应不同维度在相似性计算中的重要性差异，
恢复类似 softmax 的尖峰注意力特性。
3. 解决低秩问题：针对 SM 的低秩特性可能导致的退化解问题，探索深度 wise 和可变形卷积等技术提升秩，增强模型表
达能力，具体可参考消融实验。

三、微观设计考量
PolarFormer 模块优势明显。极性感知注意力机制恢复了负向信息，使模型能捕捉更全面的关系，增强了注意力图的判别力。
可学习幂函数有效降低了注意力熵，提升了模型区分强弱响应的能力，使其能更好地聚焦重要特征。复杂度分析表明 PolarFormer 
具有线性复杂度，在保证精度的同时提升了计算效率。消融实验验证了各组件的有效性，证明了其在多种视觉任务和长序列任务中的
良好性能。

四、适用任务
PolarFormer 适用于多种视觉任务，包括 ImageNet-1K 图像分类、COCO 目标检测与实例分割、ADE20K 语义分割，以及 Long 
Range Arena（LRA）任务。在这些任务中，PolarFormer 通过有效捕捉查询 - 键交互、恢复注意力图的尖峰特性，提升了模型性
能。在图像分类任务中提高了准确率，在检测和分割任务中显著提升了相关指标。在 LRA 任务中，PolarFormer 也取得了优异成绩
，超越了其他线性注意力模型，证明了其在不同任务中的有效性和通用性。
"""

class PolaLinearAttention(nn.Module):
    def __init__(self, dim, H, W, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, dim, H // sr_ratio, W // sr_ratio)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, alpha, kernel_size))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)

        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding.flatten(2).transpose(1, 2)
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * torch.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, self.head_dim, H, W)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


if __name__ == "__main__":
    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建测试输入张量 (batch_size, channels, height, width) / B C H W
    x = torch.randn(3, 128, 32, 32).to(device)
    # 初始化 pla 模块
    pla = PolaLinearAttention(dim=128, H=32, W=32, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                              kernel_size=5, alpha=4)
    print(pla)
    print("B站:CV缝合救星")
    pla = pla.to(device)
    # 前向传播
    output = pla(x)

    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)