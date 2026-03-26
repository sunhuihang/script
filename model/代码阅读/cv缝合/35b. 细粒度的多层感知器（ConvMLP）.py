import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

"""
CV缝合救星魔改创新2：细粒度的多层感知器（ConvMLP）。
目标：
1. 卷积MLP（ConvMLP）：采用卷积的方式结合多层感知器（MLP），让每个卷积层都能够学习到更复杂的
特征表示，同时利用 MLP 模块提升特征表达的深度。
2. 通过多层卷积感知来增强特征的表示能力，特别是在语义分割任务中，ConvMLP可以用于细化特征表示，
提升模型的对复杂场景的适应能力。
实现步骤：
1. 引入卷积多层感知器（ConvMLP）：每个卷积层后面跟一个MLP模块，以学习更加丰富的特征表示。
2. 增强特征表示能力：在特征图中应用多层感知器，进一步细化特征，增强每个位置上的特征表达。
3. 利用深层卷积+MLP的结构，提升对复杂场景下细节的感知能力。
"""
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
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

# 卷积多层感知器 (ConvMLP)
class ConvMLP(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_ratio=4):
        super(ConvMLP, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(in_dim, in_dim * mlp_ratio, kernel_size=3, padding=1)
        self.norm1 = LayerNorm(in_dim * mlp_ratio, eps=1e-6, data_format="channels_first")
        self.act1 = nn.GELU()

        # MLP的卷积处理
        self.fc1 = nn.Conv2d(in_dim * mlp_ratio, in_dim * mlp_ratio, kernel_size=1)
        self.norm2 = LayerNorm(in_dim * mlp_ratio, eps=1e-6, data_format="channels_first")
        self.act2 = nn.GELU()

        # 第二层卷积
        self.conv2 = nn.Conv2d(in_dim * mlp_ratio, out_dim, kernel_size=3, padding=1)
        self.norm3 = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 第一步卷积+激活
        x = self.norm1(self.act1(self.conv1(x)))

        # 通过MLP卷积进行特征学习
        x = self.norm2(self.act2(self.fc1(x)))

        # 第二步卷积
        x = self.norm3(self.conv2(x))

        return x

# AFE模块
class AFE(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, padding=0)
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0)
        self.ctx_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=4)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        self.enhance = ConvMLP(in_dim=dim // 2, out_dim=dim // 2, mlp_ratio=4)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.norm1(self.act(self.dwconv(x)))
        x = self.norm2(self.act(self.proj1(x)))
        ctx = self.norm3(self.act(self.ctx_conv(x)))  # SCM模块

        enh_x = self.enhance(x)  # FRM模块
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))
        return x

# AFEBlock模块
class AFEBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1, expan_ratio=4, kernel_size=3):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MLP(dim=dim, mlp_ratio=expan_ratio)
        self.attn = AFE(dim, kernel_size=kernel_size)
        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        inp_copy = x
        x = self.layer_norm1(inp_copy)
        x = self.drop_path_1(self.attn(x))
        out = x + inp_copy
        x = self.layer_norm2(out)
        x = self.drop_path_2(self.mlp(x))
        out = out + x
        return out

# MLP模块
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, use_dcn=False):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x

# 测试代码
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64)  # 随机生成一张输入图片张量
    AFEBlock = AFEBlock(dim=32)
    output = AFEBlock(input)  # 进行前向传播
    # 输出结果的形状
    print("AFEBlock_输入张量的形状：", input.shape)
    print("AFEBlock_输出张量的形状：", output.shape)
