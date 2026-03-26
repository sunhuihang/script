import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
'''
SFFNet: 基于小波的空间和频域融合网络用于遥感分割 (Arxiv 2024)
即插即用模块：WTFD 小波变换高低频特征分解模块 
一、背景：
WTFD是SFFNet网络中用于将频域信息引入遥感图像分割的一种模块。其主要目的是通过频域特征增强对复杂区域
（如阴影、边缘和纹理变化显著区域）的分割能力，弥补纯空间特征的不足。

二、WTFD工作原理：
1. 初步特征增强：首先对输入特征进行卷积操作，增加非线性表示，以便在分解过程中保留更多信息。
2. Haar小波变换分解：通过 Haar 小波变换将特征分解为四种分量：
a. 低频成分（A）（代表图像的整体信息）:低频成分代表图像中缓慢变化的部分，通常对应图像的整体轮廓、
色块和大尺度的结构信息。它包含的是图像中的平滑区域和整体的布局，忽略了细节。
b. 水平高频成分（H）（代表水平方向上的细节信息）:高频成分对应的是图像中快速变化的部分，通常包括
图像的细节、纹理、边缘等信息。高频部分可以捕捉图像中的细微变化和边缘特征。
c. 垂直高频成分（V）（代表垂直方向上的细节信息）
d. 对角高频成分（D）（代表对角方向上的细节信息）。
3. 特征映射与组合：
a. 将低频分量进一步卷积得到最终的低频特征，这些特征强化了图像的全局信息。
b. 将三种高频分量拼接后卷积成高频特征，用于增强边缘和局部细节的表示能力。
4. 最终，低频和高频特征被整合到模型中，与空间特征结合，弥补传统空间特征分割方法在复杂区域的不足。

三、适用任务：遥感语义分割，图像分割，目标检测等所有CV任务。
'''

class WTFD(nn.Module): #小波变化高低频分解模块
    def __init__(self, in_ch, out_ch):
        super(WTFD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        return yL,yH


class WTFDown(nn.Module):#小波变化高低频分解下采样模块
    def __init__(self, in_ch, out_ch):
        super(WTFDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        # return yL , yH
        return yL + yH #小波变化高低频分解下采样模块

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



