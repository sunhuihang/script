import itertools
import torch
import torch.nn.functional as F
"""
CV缝合救星创新魔改2：动态并行注意力模块（Dynamic Parallel Attention Module）
不足：
1. 多头注意力的组间隔离性：在当前的CGAttention模块中，不同的注意力头被分组并以级联的方式处理，虽然减少了计算复杂度，
但组间的信息交互相对不足，导致特征融合不够充分。
创新：
1. 动态权重生成模块：在原有的注意力机制基础上，引入动态权重生成模块，该模块根据输入特征的内容动态地调整注意力权重。
这种机制能够使模型在处理不同输入时具有自适应性，从而在复杂场景下提高模型的灵活性和特征提取能力。
2. 并行通道交互模块：为了增强通道之间的信息交互，引入并行通道交互模块，使用并行的卷积操作对不同通道进行特征提取和融合，
确保通道间的信息能够充分交流，提高整体的特征建模能力。
"""
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class CascadedGroupAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=4, attn_ratio=4, resolution=7, kernels=[5, 5, 5, 5]):
        super().__init__()
        key_dim = dim // 16
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // num_heads, self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        # 动态权重生成模块
        self.dynamic_weight_gen = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
            torch.nn.Sigmoid()
        )

        # 并行通道交互模块
        self.parallel_channel_interaction = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim // 4, bias=False)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
            attn = ((q.transpose(-2, -1) @ k) * self.scale + trainingab[i])
            attn = attn.softmax(dim=-1)  # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))

        # 动态权重生成
        dw = self.dynamic_weight_gen(x)
        x = x * dw

        # 并行通道交互
        x = self.parallel_channel_interaction(x)

        return x

if __name__ == '__main__':
    input = torch.randn(1, 64, 32, 32)
    model = CascadedGroupAttention(dim=64, resolution=32)  # resolution要求和图片大小一样
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())