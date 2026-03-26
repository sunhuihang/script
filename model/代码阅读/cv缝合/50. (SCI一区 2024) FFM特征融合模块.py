import torch
import torch.nn as nn

'''
Adualencoder crack segmentation network with Haarwavelet-based high–lowf requency 
attention (SCI 2024) Expert Systems With Applications
特征融合模块（FFM）：高效融合CNN与Transformer特征，提升裂缝分割性能
一、背景
1. 裂缝分割网络融合难题：在裂缝分割领域，CNN 提取局部信息能力强，但长距离关系建模不佳，堆叠卷积易丢失小裂缝细节；
Transformer 长距离关系建模出色，却易受复杂背景干扰。现有结合两者的网络融合方式简单，缺乏深度交互，难以发挥优
势。
2. FFM 设计初衷：为解决这些问题，FFM 被设计出来，它致力于深度融合 CNN 编码器的局部特征与 Transformer 编码器的
全局上下文特征，实现跨域融合和相关性增强，以提升网络对裂缝的分割性能，应对复杂背景下小裂缝分割的难题。
二、模块原理
1. 特征预处理与对齐：用 1×1 卷积调整 CNN 编码器中间特征维度，使其与 Transformer 编码器中间特征对齐，为融合做准备。
由于两者提取的特征存在冗余，通过通道注意力（CA）调整通道权重，减少冗余，得到优化后的特征。
2. 跨域融合与增强：
a. 跨域融合块（CFB）操作：CFB 对处理后的特征进行跨域融合。先利用深度可分离卷积（DSC）获取相关参数，让 CNN 编码器特
征的部分参数与 Transformer 编码器特征的部分参数进行多头注意力计算，反之亦然，然后拼接结果并降维，充分融合跨域特征。
b. 相关性增强机制：通过矩阵乘法对 CNN 和 Transformer 编码器特征的跨域相关性进行建模，增强重要信息传递，抑制无关特
征，提升特征相关性。
3. 融合与输出：将经过前面步骤处理的特征进行拼接，再通过特定的融合块（FFB）进一步处理，包括使用倒置深度可分离卷积（IDSC）
聚合特征和降维，以及利用深度可分离卷积（DSC）和 1×1 卷积提取有效信息，最终输出融合后的特征。

三、适用任务
1. 核心应用领域：FFM 主要应用于裂缝分割任务，在基于 CNN 和 Transformer 的裂缝分割网络中处于关键地位。它能有效融合多源特征，
增强网络对裂缝特征的感知与提取能力，提高裂缝分割的准确性和召回率，助力网络在复杂背景下精准识别和分割裂缝，为基础设施的安全
监测与维护提供重要技术支持。
2. 拓展应用潜力：在更广泛的计算机视觉任务领域，如医学图像分割、目标检测、语义分割、图像分类和图像增强等方面，FFM 所体现的跨域
特征融合与相关性增强理念具有一定通用性和借鉴意义。其通过优化特征组合提升模型性能的思路，有望在其他视觉任务中改善特征利用效率，
增强模型对复杂场景和目标的处理能力，促进计算机视觉技术在多领域的发展与应用创新。
'''


class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out


class FFM(nn.Module):
    def __init__(self, dim1):
        super().__init__()
        dim2 = dim1
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        self.concat = nn.Conv2d(dim2 * 2, dim2, 1)

        self.fusion = nn.Sequential(IDSC(dim2 * 4, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    DSC(dim2, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    nn.Conv2d(dim2, dim2, 1),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU())

    def forward(self, x, y):

        b, c, h, w = x.shape
        B, N, C = b, h * w, c
        H = W = h
        x = self.trans_c(x)

        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)

        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)
        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)
        out = torch.cat([x, y, out1, out2], dim=1)
        out = self.fusion(out)
        return out

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)

    # 初始化 FFM 模块并设置输入通道维度和输出通道维度
    FFM_module = FFM(32)
    # 将输入张量传入 FFM 模块
    output = FFM_module(input1, input2)
    # 输出结果的形状
    print("FFM_输入张量的形状：", input1.shape)
    print("FFM_输出张量的形状：", input2.shape)