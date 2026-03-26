import torch
import torch.nn as nn
from timm.models.layers import DropPath
'''
CAS-ViT: 用于高效移动应用的卷积加法自注意力视觉变压器 (Arxiv 2024)
即插即用模块：CAS卷积加法自注意力模块（替身模块）
一、背景
Vision Transformers（ViTs）由于其强大的全局上下文建模能力在计算机视觉任务中表现出色，但由于成对令牌
之间的亲和性计算和复杂的矩阵操作，其在资源受限的场景（如移动设备）上的部署面临挑战。虽然以前的研究在减
少计算开销方面取得了一定的进展，但仍不足以解决这些问题。为此，提出了CAS卷积加法自注意力模块，通过简化
令牌混合器，显著降低了计算开销，在效率和性能之间取得了平衡，尤其适用于高效移动视觉应用。

二、CAS卷积加法自注意力模块原理
1. 输入特征：与标准的Transformer注意力机制类似，使用查询（Q）、键（K）和值（V）来表示输入特征。
2. 加法相似函数：引入了一种新的加法相似函数代替传统的乘法操作，减少了计算复杂度，尤其在计算成对的
令牌亲和性时显著提高了效率。
3. 卷积特征增强：通过使用卷积操作增强了局部感知能力，同时利用通道操作和空间操作提取更丰富的特征。
4. 关键模块：
A. 空间操作：使用深度可分离卷积提取空间信息，并通过批量归一化与激活函数进行增强，最后通过sigmoid
函数得到空间注意力权重。
B. 通道操作：使用全局平均池化和卷积操作提取每个通道的重要性，增强通道之间的信息交互。
C. 深度可分离卷积（DWC）：用于计算查询和键融合后的深度特征，进一步降低计算复杂度。
5. 输出特征：经过卷积加法操作后的特征通过投影和丢弃层，最终输出增强后的特征矩阵，有效保留了全局信息
的建模能力，同时显著降低了计算量。
三、适用任务
该模块适用于图像分类、目标检测、实例分割和语义分割等计算机视觉任务。尤其适合资源受限的场景（如移动设
备），在保证计算效率的同时提供具有竞争力的性能。
'''
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)
class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)
class CAS(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out
class Mlp_CASVIT(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)

class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out

# 输入 B C H W ,  输出 B C H W
if __name__ == '__main__':
    # 创建一个输入张量，形状为 B C H W
    input = torch.randn(1, 512, 64, 64)
    # 创建 CAS 模块的实例
    model = CAS(dim=512)
    # 前向传播，获取输出
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
