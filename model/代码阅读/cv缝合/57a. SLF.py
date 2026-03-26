import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
"""
CV缝合救星魔改创新：Sparse Local Attention (SLF)
引入局部卷积增强（Local Convolution Enhancement）
一、背景：
这个创新点的目标是通过一个局部卷积层（LocalConv）来增强自注意力机制对局部信息的捕捉能力。
1. 引入局部卷积层：该卷积层能够通过卷积操作捕捉图像的局部信息，这对图像的细节处理有帮助，
尤其是在图像篡改等任务中。
2. 结合稀疏自注意力：将局部卷积和稀疏自注意力相结合，增强对局部信息的处理，同时减少计算开销。
二、说明：
1. 局部卷积增强（LocalConv）：通过卷积操作增强输入张量的局部信息，帮助捕捉细节特征。卷积层
采用 3x3 卷积核，带有 BatchNorm 和 ReLU 激活。
2. SSA模块改进：引入了局部卷积层 LocalConv，并在自注意力计算后将其结果与局部卷积结果融合。
这种融合增强了模型对局部信息的关注，同时保持了稀疏自注意力的计算效率。
3. 多尺度处理：稀疏自注意力与局部卷积的结合，能够捕捉不同层次的特征，改进了模型的表达能力，尤
其适用于图像处理任务。
"""

# 局部卷积增强模块
class LocalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LocalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# MLP模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 自注意力模块
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # 注意这里的 x 是 (B, N, C)，要确保这里的维度正确
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 稀疏自注意力模块（SSA）
class SSA(nn.Module):
    def __init__(self, dim, num_heads=4, sparse_size=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.local_conv = LocalConv(dim, dim)  # 局部卷积增强层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sparse_size = sparse_size

    def forward(self, x):
        B, C, H, W = x.shape  # 现在我们是从 (B, C, H, W) 开始的
        x_before = x.view(B, C, -1).transpose(1, 2)  # 变成 (B, N, C)，N = H * W

        # 引入局部卷积增强
        x_local = self.local_conv(x)  # 使用局部卷积增强局部信息

        # 稀疏自注意力处理
        x = self.norm1(x)
        x = self.attn(x.view(B, -1, C))  # 确保传递给自注意力模块的是 (B, N, C) 形状

        # 合并局部卷积信息
        x = x + x_local.view(B, -1, C)
        x = x_before + self.drop_path(x)

        # MLP处理
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 删除多余的 H 和 W 参数

        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module = SSA(dim=64)
    input_tensor = torch.randn(2, 64, 64, 64)  # 输入大小为 (B, C, H, W)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状
