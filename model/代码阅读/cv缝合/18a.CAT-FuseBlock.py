import torch
import torch.nn as nn
from typing import List
"""
CV缝合救星模块：CAT-FuseBlock
CAT表示Channel Attention Transformer，Fuse表示高频和低频特征的融合。
通道注意力机制：通道注意力机制能够为每个通道分配一个可学习的权重，从而动态地调整高频和低频信息在不同通道上的重要性，这样
能够在不影响空间维度的情况下增强模型的表达能力。在高频和低频分支的融合过程中，引入通道注意力机制，让模型可以学习不同通道
的重要性，从而自动调整高频和低频特征在每个通道上的贡献。
"""

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.act_block(x)


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, dim, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, max(1, dim // reduction), 1, bias=False),  # 确保至少有1个通道
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, dim // reduction), dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道权重
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        return x * attn


class CloMSFM(nn.Module):
    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)

        # 通道注意力模块
        self.channel_attn = ChannelAttention(dim)

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        b, c, h, w = x.size()
        qkv = to_qkv(x)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()
        q, k, v = qkv
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module):
        b, c, h, w = x.size()
        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()
        kv = to_kv(x).view(b, 2, -1, self.dim_head, h * w).permute(1, 0, 2, 4, 3).contiguous()
        k, v = kv
        attn = self.scalor * torch.matmul(q, k.transpose(-1, -2))
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        res = torch.matmul(attn, v).transpose(2, 3).reshape(b, -1, h, w)
        return res

    def forward(self, x: torch.Tensor):
        res_high = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res_high.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))

        res_high = torch.cat(res_high, dim=1) if res_high else torch.zeros_like(x)
        res_low = self.low_fre_attention(x, self.global_q, self.global_kv)

        # 使用通道注意力模块进行高频和低频特征的加权融合
        combined = torch.cat((res_high, res_low), dim=1)
        combined = self.channel_attn(combined)
        return x + self.proj_drop(self.proj(combined))


# 测试改进后的模块
if __name__ == '__main__':
    model = CloMSFM(dim=64, num_heads=8, group_split=[4, 4], kernel_sizes=[3], window_size=7)
    input = torch.randn(1, 64, 56, 56)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
