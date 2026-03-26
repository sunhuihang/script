import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
"""
CV缝合救星魔改创新：MSFR (Multi-Scale Feature Refinement)：多尺度和特征细化。
问题：
1. 传统的特征细化模块（FRM）使用单一尺度的卷积核，缺乏灵活性，难以处理尺度变化较大的对象。
2. 模型对不同大小目标的感知力不足，无法同时捕获大范围和细节信息，导致分割精度下降。
创新改进：
1. 多尺度卷积核引入：
A. 增加多个卷积尺度：小尺度卷积（3x3）、中尺度卷积（5x5）、大尺度卷积（7x7）。
B. 通过不同尺度的卷积，补充模型对不同大小目标的感知能力。
2. 多尺度特征融合：
A. 拼接不同尺度特征：将不同尺度的卷积特征图进行拼接，保持不同尺度的信息。
B. 1x1卷积融合：使用1x1卷积融合特征，减少计算量的同时增强特征表达能力。
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

# 特征细化模块：结合多尺度特征
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        # 使用不同尺度的卷积核
        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.mconv = nn.Conv2d(in_dim, in_dim, kernel_size=5, stride=1, padding=2, groups=in_dim)  # 中等尺度卷积
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=7, stride=1, padding=3, groups=in_dim)  # 大尺度卷积

        # 每个卷积后的规范化
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")

        # 激活函数
        self.act = nn.GELU()

        # 下采样操作
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                              groups=in_dim)
        # 1x1 卷积用于特征融合
        self.proj = nn.Conv2d(in_dim * 3, out_dim, kernel_size=1, stride=1, padding=0)  # 改为3种尺度的拼接

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
        B, C, H, W = x.shape

        # 下采样
        dx = self.down(x)
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)

        # 使用不同尺度的卷积核进行处理
        lx = self.norm1(self.lconv(self.act(x * udx)))  # 小尺度
        mx = self.norm2(self.mconv(self.act(x * udx)))  # 中尺度
        hx = self.norm3(self.hconv(self.act(x * udx)))  # 大尺度

        # 将不同尺度的特征拼接
        out = self.act(self.proj(torch.cat([lx, mx, hx], dim=1)))
        return out

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

        self.enhance = FeatureRefinementModule(in_dim=dim // 2, out_dim=dim // 2, down_kernel=3, down_stride=2)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.norm1(self.act(self.dwconv(x)))
        x = self.norm2(self.act(self.proj1(x)))
        ctx = self.norm3(self.act(self.ctx_conv(x)))  # SCM模块

        enh_x = self.enhance(x)  # FRM模块
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))
        return x

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
