import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple
"""
CV缝合救星魔改创新1：引入深度可分离卷积
一、背景：
在图像分割和计算机视觉任务中，传统的卷积操作和注意力机制常常面临高计算开销和效率瓶颈，特别是对于高分辨率输入图像时。
为了解决这个问题，提出了深度可分离卷积。通过动态调整注意力区域并优化卷积计算，能够在减少计算量的同时提升特征提取能力，
增强前景定位精度。
二、创新点：
深度可分离卷积优化：
深度可分离卷积（Depthwise Separable Convolution）被引入来替代传统卷积操作。它通过将卷积操作分解为深度卷积和逐
点卷积，大大减少了参数数量和计算量，从而提升了计算效率，特别是在处理高分辨率输入时。
"""
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    """使用深度可分离卷积减少计算复杂度"""
    def __init__(self, inp, outp, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp)
        self.pointwise = nn.Conv2d(inp, outp, kernel_size=1, stride=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DynamicAttention(nn.Module):
    """ 轻量化动态注意力机制，改进了RCA机制的计算效率 """
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=2, relu=True):
        super(DynamicAttention, self).__init__()
        self.dwconv_hw = DepthwiseSeparableConv(inp, inp, square_kernel_size, padding=square_kernel_size // 2)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        # 动态自适应权重计算
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)
        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)  # 使用深度可分离卷积提取特征
        att = self.sge(x)  # 计算动态注意力权重
        out = att * loc  # 动态调整后的输出
        return out


class RCM(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=DynamicAttention,  # 使用改进后的动态注意力机制
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


# 输入 N C H W, 输出 N C H W
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64)  # 随机生成一张输入图片张量
    rcm = RCM(dim=32)  # 初始化RCM模块
    output = rcm(input)  # 进行前向传播
    print("RCM_输入张量的形状：", input.shape)
    print("RCM_输出张量的形状：", output.shape)
