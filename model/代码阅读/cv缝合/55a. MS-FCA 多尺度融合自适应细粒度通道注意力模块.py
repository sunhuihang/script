import math
import torch
from torch import nn
"""
CV缝合就行魔改创新：MS-FCA多尺度融合自适应细粒度通道注意力模块
一、FCA不足
仅专注于在固定的单一尺度下利用相关矩阵捕捉全局和局部信息关联并分配权重，缺乏对多尺度特征的考虑。
在实际图像中，不同物体或区域的特征可能在不同尺度下表现各异，单一尺度的处理方式限制了其对图像信
息的全面理解和有效利用，在面对复杂背景或多尺度目标时，可能无法准确地突出重要特征和抑制无关信息，
进而影响了图像去雾等任务的准确性和效果。
二、改进创新点：
原模块在处理图像特征时，仅从单一尺度对全局和局部信息进行提取与融合，难以全面捕捉图像中不同尺度
下的特征差异。而 MS - FCA 模块引入了多尺度特征融合机制，通过对输入特征图进行不同尺度的下采样
和上采样操作，能够在多个尺度上应用 FCA 模块处理并融合结果，从而增强了对不同尺度特征的感知和利
用能力，有望更好地适应复杂图像场景，提升图像去雾等任务的性能。
"""
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class MultiScaleFCAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2, num_scales=3):
        super(MultiScaleFCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num_scales = num_scales
        self.fc_attention_modules = nn.ModuleList()
        for _ in range(num_scales):
            self.fc_attention_modules.append(FCAttention(channel, b, gamma))

        # 用于融合不同尺度特征的卷积层
        self.fusion_conv = nn.Conv2d(channel * num_scales, channel, 1, padding=0, bias=True)

    def forward(self, input):
        outputs = []
        for scale in range(self.num_scales):
            # 进行不同尺度的下采样和上采样操作
            if scale == 0:
                x = input
            elif scale == 1:
                x = nn.functional.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
            else:
                x = nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)

            # 应用 FCA 模块处理
            x = self.fc_attention_modules[scale](x)

            # 恢复到原始尺寸
            if scale!= 0:
                x = nn.functional.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=False)

            outputs.append(x)

        # 融合不同尺度的特征
        out = torch.cat(outputs, dim=1)
        out = self.fusion_conv(out)

        return input * out


class FCAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)  # (1,1,64)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)  # (1,64,1,1)
        # x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)

        # out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input * out


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 64, 256, 256)
    model = MultiScaleFCAttention(channel=64)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())