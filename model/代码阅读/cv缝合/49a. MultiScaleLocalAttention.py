import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3
"""
CV缝合救星魔改创新1：MultiScaleLocalAttention
一、原理
1. 仅使用单一尺度的特征图进行注意力计算可能会丢失一些重要的信息。通过融合多尺度特征，可以让模块捕捉到不同层次的特征信息，
从而提升注意力机制的效果。
2. 例如，在超分辨率任务中，不同尺度的特征包含了不同程度的细节和结构信息，融合这些信息有助于更好地重建图像。
二、实现方法
1. 在LocalAttention类中添加多尺度特征提取的模块。可以通过引入不同步长的卷积层或者池化层来获取不同尺度的特征图。
2. 设计一个融合模块，将不同尺度的特征图进行融合。例如，使用逐点相加、拼接后再接卷积层等方式来融合特征。
"""

class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class MultiScaleLocalAttention(nn.Module):
    ''' attention based on local importance with multi - scale feature fusion'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.scale1_body = nn.Sequential(
            # sample importance for scale 1
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.scale2_body = nn.Sequential(
            # sample importance for scale 2
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(5, stride=2),
            nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map for scale 1
        g1 = self.gate(x[:, :1].clone())
        w1 = F.interpolate(self.scale1_body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # interpolate the heat map for scale 2
        g2 = self.gate(x[:, :1].clone())
        w2 = F.interpolate(self.scale2_body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # fusion of multi - scale features
        w = (w1 + w2) / 2
        g = (g1 + g2) / 2
        return x * w * g


if __name__ == "__main__":
    input = torch.randn(1, 32, 64, 64)
    MLA = MultiScaleLocalAttention(32)
    output = MLA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())