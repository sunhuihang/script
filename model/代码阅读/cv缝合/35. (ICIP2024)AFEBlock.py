import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
'''
自适应特征增强（AFE）模块：用于语义分割任务的特征增强模块（ICIP 2024）
即插即用模块：AFE（替身模块）
一、背景
在语义分割任务中，传统卷积神经网络（CNN）方法受限于固定几何结构，难以获取长距离上下文信息；基于视觉 Transformer
的方法虽利用自注意力机制取得一定进展，但存在对语义级上下文信息捕捉不足、全局感受野主导导致难以捕捉细节、数据需求大
等问题；混合注意力模型在处理复杂背景下的分割对象时仍面临困难，且对于半透明对象的分割也是一大挑战。为应对这些问题，
本文提出了 AFE 模块，以更好地处理复杂场景中的语义分割任务。

二、AFE 模块原理
1. 输入特征：接收经过归一化和卷积嵌入处理后的特征，该特征在一定程度上学习了泛化和判别能力，且通道已被压缩一半以减少
计算量并促进特征混合。
2. 模块组成与处理：
A. 空间上下文模块（SCM）：采用较大核（7x7）的分组卷积，旨在增加感受野，从而能够在更大范围内捕捉空间上下文信息，
以应对场景中的尺度变化。
B. 特征细化模块（FRM）：受图像锐化和对比度增强概念启发，负责生成语义线索，通过下采样、上采样、特征差异计算和元素级乘
法等操作，捕获低频和高频区域特征。
3. 输出融合与增强：SCM 和 FRM 的输出先通过 1x1 卷积层进行融合，然后再经过卷积多层感知器（ConvMLP）进一步增强特征
表示，使模型能够更好地处理杂乱背景下的语义分割任务。
三、适用任务
1. 语义分割任务：尤其适用于存在背景杂乱、半透明对象以及尺度变化等挑战的场景，如 ZeroWaste-f 数据集中的垃圾分割任务，
可准确区分和分类各种复杂背景下的垃圾材料。
2. 其他复杂场景下的目标分割、检测任务：通过有效整合空间上下文信息和语义线索，能够提升在复杂环境中对目标物体的分割准确性，
使对象边界更清晰易区分。
3. 适用于其他所有CV任务。
'''

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
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                              groups=in_dim)
        self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)

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

        dx = self.down(x)
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))

        out = self.act(self.proj(torch.cat([lx, hx], dim=1)))

        return out
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
        ctx = self.norm3(self.act(self.ctx_conv(x)))  #SCM模块

        enh_x = self.enhance(x)                       #FRM模块
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))
        return x
class AFEBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1, expan_ratio=4,kernel_size=3, use_dilated_mlp=True):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        if use_dilated_mlp:
            self.mlp = AtrousMLP(dim=dim, mlp_ratio=expan_ratio)
        else:
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
class AtrousMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos1 = nn.Conv2d(dim * mlp_ratio, dim * 2, 3, padding=1, groups=dim * 2)
        self.pos2 = nn.Conv2d(dim * mlp_ratio, dim * 2, 3, padding=2, dilation=2, groups=dim * 2)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.act(self.fc1(x))
        x1 = self.act(self.pos1(x))
        x2 = self.act(self.pos2(x))
        x_a = torch.cat([x1, x2], dim=1)
        x = self.fc2(x_a)

        return x

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64) #随机生成一张输入图片张量
    # 初始化AFEBlock模块并设定通道维度
    AFEBlock  = AFEBlock(dim=32)
    output = AFEBlock(input)  # 进行前向传播
    # 输出结果的形状
    print("AFEBlock_输入张量的形状：", input.shape)
    print("AFEBlock_输出张量的形状：", output.shape)
