import torch
import torch.nn as nn
from einops import rearrange
"""
CV缝合救星魔改创新2：动态稀疏注意力
一、不足
目前的TKSA 模块使用固定的 top-k 稀疏策略，尽管支持不同比例的稀疏度，但这些比例是硬编码的，无法根据输入特征的不同自适应调整。
设计了一个动态稀疏性策略，让模块根据特征信息的复杂性动态决定 k 的选择。
二、改进亮点
1. 动态稀疏性：使用 dynamic_k 模块，根据输入特征自适应调整每个注意力头的稀疏度，避免固定比例带来的局限性。
2. 更高的灵活性：每个注意力头的 top-k 操作动态变化，能够更好地适应不同特征分布。
"""
class DynamicTKSA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(DynamicTKSA, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dynamic_k = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, num_heads, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Generate Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # Compute dynamic top-k sparsity for each head
        k_values = self.dynamic_k(x).view(b, self.num_heads, 1, 1)
        k_values = (k_values * attn.size(-1)).long().clamp(1, attn.size(-1))

        sparse_attn = []
        for head_idx in range(self.num_heads):
            # 动态调整 top-k 稀疏性
            k_values_for_head = k_values[:, head_idx, 0, 0]  # 提取当前 head 的 k 值 (b,)
            attn_head = attn[:, head_idx]  # 当前 head 的注意力 (b, seq_len, seq_len)
            mask = torch.zeros_like(attn_head)

            for batch_idx in range(b):
                k_value = k_values_for_head[batch_idx].item()
                topk_indices = torch.topk(attn_head[batch_idx], k=k_value, dim=-1)[1]
                mask[batch_idx].scatter_(-1, topk_indices, 1.0)

            sparse_attn.append(attn_head * mask)

        attn = torch.stack(sparse_attn, dim=1).softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

if __name__ == "__main__":
    input_tensor = torch.randn(5, 32, 64, 64)
    dynamic_tksa = DynamicTKSA(32)
    output = dynamic_tksa(input_tensor)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())
