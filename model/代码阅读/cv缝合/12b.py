import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积 (Depthwise Convolution)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)

        # 逐点卷积 (Pointwise Convolution)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)  # 深度卷积
        x = self.pointwise(x)  # 逐点卷积
        return x


class WTFD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WTFD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        # 低频特征处理 (普通卷积)
        self.conv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # 高频特征处理（深度可分离卷积）
        self.conv_bn_relu_H = nn.Sequential(
            DepthwiseSeparableConv(in_ch * 3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 小波变换
        yL, yH = self.wt(x)

        # 提取高频的三个部分
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        # 将三个高频部分拼接
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)

        # 处理低频特征
        yL = self.conv_bn_relu_L(yL)

        # 处理高频特征（使用深度可分离卷积）
        yH = self.conv_bn_relu_H(yH)

        return yL, yH

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1,32, 64, 64)
    # 创建一个 WTFD实例
    WTFD =  WTFD(32,64)
    # 将输入特征图传递给 WTFD模块
    output_L,output_H = WTFD(input) #小波变化高低频分解模块
    # 打印输入和输出的尺寸
    print(f"input  shape: {input.shape}")
    print(f"output_L shape: {output_L.shape}")
    print(f"output_H shape: {output_H.shape}")