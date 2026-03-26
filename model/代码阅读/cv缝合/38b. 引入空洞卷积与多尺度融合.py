import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple
"""
CV缝合救星魔改创新2：引入空洞卷积与多尺度融合
推荐模块名称：Dilated Multi-Scale Fusion Module" (DMSFM)
一、背景：
在许多视觉任务（如语义分割、目标检测等）中，捕捉不同尺度的上下文信息对于提升模型的性能至关重要。
传统的卷积操作通常受到卷积核大小的限制，而空洞卷积（Dilated Convolution）能够通过增加卷积核
之间的间距来增加感受野，从而在不增加计算量的情况下，获得更丰富的上下文信息。此外，通过多尺度特
征融合，模型可以在不同尺度上处理特征，进一步提升精度，尤其是在处理包含多尺度物体的复杂场景时。
二、创新点：
1.空洞卷积（Dilated Convolution）：空洞卷积引入了卷积核间的间距，从而增大感受野，帮助网络捕
捉更大范围的上下文信息，而不增加计算量。
2. 多尺度特征融合：通过结合来自不同尺度的特征（例如通过不同尺寸的卷积核和池化操作），增强模型
对图像中多种尺度物体的感知能力。这有助于提升前景与背景的分割精度，尤其是在目标大小变化较大的场景中。
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
    """ 使用深度可分离卷积减少计算复杂度 """
    def __init__(self, inp, outp, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp)
        self.pointwise = nn.Conv2d(inp, outp, kernel_size=1, stride=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DilatedConvBlock(nn.Module):
    """ 空洞卷积块，用于扩大感受野并融合多尺度信息 """
    def __init__(self, inp, outp, dilation_rate=2, kernel_size=3, padding=2):
        super(DilatedConvBlock, self).__init__()
        self.dilated_conv = nn.Conv2d(inp, outp, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.dilated_conv(x))


class MultiScaleFusion(nn.Module):
    """ 多尺度特征融合模块 """

    def __init__(self, inp, outp):
        super(MultiScaleFusion, self).__init__()
        self.conv1 = DilatedConvBlock(inp, outp, dilation_rate=1)  # 标准卷积
        self.conv2 = DilatedConvBlock(inp, outp, dilation_rate=2)  # 空洞卷积（dilation = 2）
        self.conv3 = DilatedConvBlock(inp, outp, dilation_rate=4)  # 空洞卷积（dilation = 4）
        self.conv4 = nn.Conv2d(inp * 3, outp, kernel_size=1)  # 融合后的1x1卷积

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # 使用AdaptiveAvgPool2d对每个张量进行尺寸对齐，确保它们的空间尺寸相同
        x1 = nn.functional.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x3 = nn.functional.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)

        # 融合不同尺度的特征
        fused = torch.cat([x1, x2, x3], dim=1)  # 拼接
        return self.conv4(fused)


class RCM(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=MultiScaleFusion,  # 使用多尺度融合模块
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
        self.token_mixer = token_mixer(dim, dim)
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
