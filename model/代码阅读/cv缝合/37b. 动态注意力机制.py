import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新2：动态注意力机制
一、背景：引入一个动态注意力机制（Dynamic Attention Mechanism, DAM）。这个机制会根据输入特征
自适应地调整注意力权重，而不是使用传统的固定的注意力权重或预定义的卷积核大小。该创新点的主要
思想是：网络能够根据当前输入图像的局部和全局特征信息，自动决定不同区域的关注程度。这种机制能够
更智能地选择重要区域进行特征强化，避免无效的计算和过度拟合。
二、创新点：
1. 动态注意力机制（DAM）：通过计算每个特征的自适应权重，使得网络可以灵活地关注图像的关键区域。
利用这种自适应的权重机制，不仅提升了特征的表示能力，还使得网络能够聚焦在最相关的区域。
2. 动态空间注意力与通道注意力：结合空间和通道的动态调整，通过对不同特征维度的自适应关注来实现特
征的精细化提取。
3. 提升模型表达能力：通过自适应权重，模型能够更智能地从不同的特征区域中提取信息，从而提高重建图
像的质量和细节恢复。
"""
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
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
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)
        return x * self.scale + shortcut


class DynamicAttention(nn.Module):
    def __init__(self, n_feats, attn_type='both', attention_scale=2.0):
        super().__init__()
        self.attn_type = attn_type
        self.attention_scale = attention_scale

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // 8, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(n_feats // 8, n_feats, 1, 1, 0),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(n_feats, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attn = self.channel_attention(x)
        spatial_attn = self.spatial_attention(x)

        if self.attn_type == 'both':
            # Combine both spatial and channel attention
            x = x * (1 + self.attention_scale * channel_attn * spatial_attn)
        elif self.attn_type == 'channel':
            x = x * (1 + self.attention_scale * channel_attn)
        elif self.attn_type == 'spatial':
            x = x * (1 + self.attention_scale * spatial_attn)

        return x


class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.LKA5 = nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats)
        self.LKA3 = nn.Conv2d(n_feats, n_feats, 3, 1, 3 // 2, groups=n_feats)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.LKA7(x) + self.LKA5(x) + self.LKA3(x)
        return x * self.scale + shortcut


class MAB(nn.Module):
    def __init__(self, n_feats, attn_type='both', attention_scale=2.0):
        super().__init__()
        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)
        self.DAM = DynamicAttention(n_feats, attn_type, attention_scale)

    def forward(self, x):
        # Process with large kernel attention
        x = self.LKA(x)
        # Extract local features with GSAU
        x = self.LFE(x)
        # Apply dynamic attention mechanism
        x = self.DAM(x)
        return x


if __name__ == "__main__":
    input = torch.randn(1, 30, 128, 128)
    MAB_model = MAB(30)
    output = MAB_model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
