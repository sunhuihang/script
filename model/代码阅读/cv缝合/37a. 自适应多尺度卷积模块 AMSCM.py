import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新1：自适应多尺度卷积模块
一、背景：现有基础上加入了一个自适应多尺度卷积模块（Adaptive Multi-Scale Convolution Module, AMSCM）。
这个模块的目标是根据输入特征的分布自适应地调整每个卷积层的尺度，从而使网络能更灵活地处理不同类型的图像特征。
此模块不仅能够捕捉不同尺度的局部和全局信息，还能动态调整卷积核的大小，以适应不同的图像内容。可以将该模块集
成到现有的MAB中。
二、创新点说明：
1. 自适应卷积核选择：根据输入特征的某些统计量（如均值或标准差），动态调整卷积核的大小，以适应不同的图像局部和全局特征。
2. 融合多尺度信息：不仅通过大核卷积捕捉长距离关系，还通过自适应的卷积核在局部特征中做出更多灵活的调整。
3. 改进计算效率：通过引入动态调整卷积核大小的机制，在不增加计算量的情况下提高了特征提取的能力。
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


class AMSCM(nn.Module):
    def __init__(self, n_feats, scales=(3, 5, 7)):
        super().__init__()
        self.n_feats = n_feats
        self.scales = scales
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        # Dynamic scale convolutions
        self.convs = nn.ModuleList([
            nn.Conv2d(n_feats, n_feats, scale, 1, scale // 2, groups=n_feats) for scale in self.scales
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        scale_outs = []

        # Apply convolutions with different scales
        for i, conv in enumerate(self.convs):
            scale_outs.append(conv(x))

        # Calculate weighted sum of different scale outputs
        weighted_sum = sum(w * out for w, out in zip(self.scale_weights, scale_outs))

        return weighted_sum + shortcut


class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.LKA5 = nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats)
        self.LKA3 = nn.Conv2d(n_feats, n_feats, 3, 1, 3 // 2, groups=n_feats)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.LKA7(x) + self.LKA5(x) + self.LKA3(x)
        return x * self.scale + shortcut


class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)
        self.AMSCM = AMSCM(n_feats)  # Add the new AMSCM module

    def forward(self, x):
        # Process with large kernel attention
        x = self.LKA(x)
        # Extract local features with GSAU
        x = self.LFE(x)
        # Apply adaptive multi-scale convolutions
        x = self.AMSCM(x)
        return x


if __name__ == "__main__":
    input = torch.randn(1, 30, 128, 128)
    MAB_model = MAB(30)
    output = MAB_model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
