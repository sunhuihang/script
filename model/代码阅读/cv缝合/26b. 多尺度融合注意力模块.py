import torch
import torch.nn as nn
from timm.models.layers import DropPath
"""
CV缝合救星创新魔改2：多尺度融合注意力模块（Multi-Scale Fusion Attention Module）
背景：
1. 现有的卷积操作通常只能处理单一尺度的特征，这可能导致对不同尺度下的图像特征缺乏敏感度，限制了模型的表现。
2. 复杂背景和细节的捕获对于高分辨率视觉任务至关重要，单尺度的处理方式难以全面捕捉这些信息。
创新：
引入多尺度融合注意力模块，通过多个不同尺度的卷积核提取特征，并将这些特征融合，以捕捉图像中的全局和局部信息，提高模型的特征建模能力。
"""
class MultiScaleFusionAttention(nn.Module):
    def __init__(self, dim, scales=[3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=s, stride=1, padding=s//2, groups=dim)
            for s in scales
        ])
        self.proj = nn.Conv2d(dim * len(scales), dim, 1)

    def forward(self, x):
        multi_scale_features = [conv(x) for conv in self.convs]
        fused_features = torch.cat(multi_scale_features, dim=1)
        out = self.proj(fused_features)
        return out

class CASWithMultiScaleFusion(nn.Module):
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
        self.multi_scale_fusion = MultiScaleFusionAttention(dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        qk_sum = q + k
        qk_sum = self.multi_scale_fusion(qk_sum)
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
    # 创建带有多尺度融合注意力模块的CAS模块的实例
    model = CASWithMultiScaleFusion(dim=512)
    # 前向传播，获取输出
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
