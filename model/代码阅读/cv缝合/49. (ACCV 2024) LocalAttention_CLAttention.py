import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3

'''
PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution(ACCV 2024)
即插即用模块:LIA（替身模块）
一、背景
1. 超分辨率网络注意力机制困境
在超分辨率研究中，降低延迟至关重要，但现有注意力机制存在问题。一阶注意力（如 ESA）性能提升有限，
二阶注意力（如自注意力、非局部注意力）虽能长程建模，但因二次复杂度在低延迟 SR 模型中应用受限。
2. LIA 的目标
LIA 旨在通过简单操作实现二阶信息交互，平衡计算与性能，解决现有注意力机制在超分辨率网络中的不足，
提升模型性能与效率，降低延迟。
二、模块原理
1. 输入与初步处理
接收输入特征图，以降低分辨率的特征图为基础进行操作。通过堆叠 softpool 和 3×3 卷积计算像素局部重要性，
利用 stride 和 squeeze 卷积减少计算、扩大感受野，采用 sigmoid 和 bilinear 进行激活和重缩放。
2. 核心计算与融合机制
计算像素在邻域内的局部重要性，采用输入的第一个通道图作为门机制重新校准局部重要性，避免相关操作带来的伪影，
按特定公式进行最终计算。
三、适用任务
1. 主要适用于超分辨率任务，在提升超分辨率模型性能和降低延迟方面起关键作用，通过改进注意力机制促进特征处理
与信息交互，提高图像重建质量。
2. 适用任务：医学图像分割，目标检测，语义分割，图像分类，图像增强等所有计算机视觉CV任务通用的模块。
'''

class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool
class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g  # (w + g) #self.gate(x, w)

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


if __name__ == "__main__":
    input = torch.randn(1, 32, 64, 64)
    LA = LocalAttention(32)
    output = LA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

