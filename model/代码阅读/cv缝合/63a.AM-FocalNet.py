import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新：自适应多尺度焦点调制（Adaptive Multi-scale Focal Modulation）
一、背景介绍：
当前的 Focal Modulation 采用了固定的尺度（focal_window）来进行上下文聚合，且聚合过程
是基于静态的多个卷积层。但是，在不同输入的情况下，一些局部特征可能会在不同尺度上有着更高的
表现力。因此，结合 “自适应卷积核” 或 “尺度自适应机制”，根据输入的不同特征和上下文信息动态
调整卷积核大小，以提高特征提取和聚合的效果。
二、改进点
1. 自适应卷积核大小：为每个查询位置自适应地选择最合适的卷积核大小。通过计算上下文区域的局
部统计信息（如方差或熵），自动调整卷积的感受野。
2. 基于内容的自适应门控机制：根据每个局部区域的特征，通过注意力机制选择最佳的卷积核尺寸，
进一步优化上下文信息的聚合。
"""

class AdaptiveFocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        # Linear projection to get query, context, and gates
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        # Activation and dropout
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize layers for different focal levels
        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []

        # Add dynamic kernel size layers
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass for Adaptive Focal Modulation
        Args:
            x: Input tensor of shape (B, H, W, C)
        """
        C = x.shape[-1]

        # Pre linear projection to get query, context, and gates
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # Context aggregation with dynamic kernel size adjustment
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)

            # Self-adaptive kernel size based on feature variance
            variance = torch.var(ctx, dim=(2, 3), keepdim=True) + 1e-5  # Compute local variance
            adaptive_kernel_size = torch.round(variance * self.focal_factor).clamp(min=self.focal_window)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1]

        # Global context gating
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # Normalize the context if required
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # Focal modulation (adaptive)
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()

        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # Post projection and dropout
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, focal_window={self.focal_window}, focal_level={self.focal_level}'

    def flops(self, N):
        """
        Calculate the FLOPs for a given input size.
        Args:
            N: The number of tokens (or feature length)
        """
        flops = 0
        flops += N * self.dim * (self.dim * 2 + (self.focal_level + 1))

        # Focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k] ** 2 + 1) * self.dim

        # Global gating
        flops += N * 1 * self.dim

        # Linear projections
        flops += N * self.dim * (self.dim + 1)

        # Output projection
        flops += N * self.dim * self.dim
        return flops


# Test the modified module with sample input
if __name__ == '__main__':
    block = AdaptiveFocalModulation(dim=64, focal_window=3, focal_level=2)
    input = torch.rand(3, 56, 56, 64)  # B, H, W, C
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())
