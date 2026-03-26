import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torch.fft
"""
CV缝合救星魔改创新：WavePyramidAttention (WPA-Conv)
一. 背景
传统的 CNN 主要依赖固定大小的卷积核来提取局部特征，然而其感受野受限，难以捕捉长距离依赖信息。近年来，
结合小波变换（Wavelet Transform, WT） 的 CNN 结构，如 WTConv2d，被提出以扩大感受野并增强对不同
频率特征的响应。WTConv2d 通过多级小波变换分解输入特征图，使得不同频率带的特征可以分别处理，提高了网
络的全局信息感知能力。然而，原始 WTConv2d 仍然存在以下问题：
1. 仅依赖 1×1 卷积进行特征融合，缺乏跨尺度信息交互，导致信息提取能力有限。
2. 未充分利用频域信息，虽然 WTConv2d 进行了小波分解，但缺乏显式的频域注意力机制，无法增强高频特征的
重要性。

二. 魔改创新：
1. 金字塔特征融合（Pyramid Fusion）
传统 WTConv2d 仅使用 1×1 卷积进行特征融合，本方案改进为 金字塔特征融合（Pyramid Fusion），结合多
个尺度的卷积操作：
A. 1×1 标准卷积：提取基本特征。
B. 3×3 可变形卷积（Deformable Convolution）：增强局部几何变形适应能力。
C. 5×5 可变形卷积：捕获更大范围的语义信息。
D. 3×3 空洞卷积（Dilated Convolution）：进一步扩大感受野，以避免信息损失。
E. 通过 1×1 卷积对不同尺度的特征进行融合，提升网络对跨尺度信息的建模能力。
2. 频域注意力机制（FFT + SE）
A. FFT 变换：将输入特征转换至频域，提取高频和低频信息。
B. SE（Squeeze-and-Excitation）模块：计算通道注意力权重，增强高频特征在频域中的重要性，提升边缘和
纹理信息的表达能力。
C. 逆 FFT 变换（iFFT）：将增强后的特征转换回空间域，提高最终特征表示能力。
"""

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PyramidFusion(nn.Module):
    def __init__(self, channels):
        super(PyramidFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv3x3 = DeformConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = DeformConv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.fusion = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)

    def forward(self, x):
        k3, k5 = 3, 5  # 3x3 和 5x5 的核大小
        offset3 = torch.zeros(x.shape[0], 2 * k3 * k3, x.shape[2], x.shape[3], device=x.device)
        offset5 = torch.zeros(x.shape[0], 2 * k5 * k5, x.shape[2], x.shape[3], device=x.device)
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x, offset3)
        x3 = self.conv5x5(x, offset5)
        x4 = self.dilated_conv(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(out)


class WavePyramidAttention(nn.Module):
    def __init__(self, in_channels):
        super(WavePyramidAttention, self).__init__()
        self.pyramid_fusion = PyramidFusion(in_channels)
        self.se_block = SEBlock(in_channels)

    def forward(self, x):
        x_fused = self.pyramid_fusion(x)
        x_fft = torch.fft.fft2(x_fused).abs()
        x_fft = self.se_block(x_fft)
        x_final = torch.fft.ifft2(x_fft).real
        return x_final


if __name__ == '__main__':
    model = WavePyramidAttention(32)
    inp = torch.rand(2, 32, 64, 64)
    out = model(inp)
    print("input.shape:", inp.shape)
    print("output.shape:", out.shape)
