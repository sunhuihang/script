import torch
import torch.nn as nn
from timm.models.layers import DropPath
"""
CV缝合救星魔改创新1：混合门控机制（Hybrid Gating Mechanism）
背景：
1. 在现有的自注意力机制中，特征选择能力存在一定的局限性，缺乏灵活性，无法根据输入特征的动态变化进行有效调整。
2. 传统的特征增强方式缺少对特征重要性的精准辨别，可能导致模型对冗余信息的关注。
创新：
引入混合门控机制，通过卷积和Sigmoid激活生成门控权重，动态调整不同特征的贡献度，提高模型对有效特征的选择性。
"""
class HybridGatingMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gated = self.gate(x)
        return x * gated

class CASWithHybridGate(nn.Module):
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
        self.hybrid_gate = HybridGatingMechanism(dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        qk_sum = q + k
        qk_sum = self.hybrid_gate(qk_sum)
        out = self.proj(qk_sum * v)
        out = self.proj_drop(out)
        return out

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

if __name__ == '__main__':
    # 创建一个输入张量，形状为 B C H W
    input = torch.randn(1, 512, 64, 64)
    # 创建带有混合门控机制的CAS模块的实例
    model = CASWithHybridGate(dim=512)
    # 前向传播，获取输出
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
