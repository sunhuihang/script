import torch
import torch.nn as nn
'''
CMUNeXt: 基于大卷积核和跳跃融合的高效医学图像分割网络 (2024 ISIB)
一、研究目的和动机：
1. U形架构（如U-Net）在医学图像分割中非常流行，但由于传统卷积的局部特性，难以有效提取全局上下文信息。
2. 现有的混合CNN-Transformer架构虽然能够提取更好的全局上下文，但需要大量计算资源。
3. CMUNeXt旨在在高分割准确性和低计算需求之间取得平衡，使其适合在移动设备上进行实时诊断。

二、CMUNeXt网络设计：
1. CMUNeXt 是一个完全卷积的网络，利用大卷积核和倒置瓶颈设计来高效捕获全局上下文信息。
大卷积核
    a. 深度分离卷积是一种轻量级的卷积操作，它将标准卷积分解为两个部分：深度卷积和逐点卷积。
    b.深度卷积（Depthwise Convolution）：对输入的每个通道分别执行卷积操作，相当于单独对每个通
    道进行空间信息提取，从而降低了计算量。
    c.逐点卷积（Pointwise Convolution）：使用1x1卷积对每个位置进行通道间的信息融合，允许各通道
    之间的特征相互交互。
    在CMUNeXt模块中，使用了大卷积核的深度分离卷积来扩展感受野，以此来捕获更远的空间上下文信息。
    大卷积核可以覆盖较大范围的像素，从而更好地获取图像的全局信息，而不是只关注局部特征。
倒置瓶颈
    a. 在CMUNeXt模块中，倒置瓶颈的设计包括在两层逐点卷积之间设置一个四倍扩展的隐藏维度。这意味着中间特征
    的通道数是输入特征的四倍，从而更全面地混合了空间和通道信息。
    b. 这种设计增加了特征空间的容量，使得大卷积核提取到的全局空间信息可以在倒置瓶颈中进行充分的通道混合。
2. 引入了新的跳跃融合（Skip-Fusion）块，用于跳跃连接，促进平滑的特征融合，从而增强从编码器
到解码器阶段的特征传递。
3. 网络采用五级编码器-解码器结构，通过深度可分离卷积减少冗余参数，同时保持出色的性能。

三、适用于：医学图像分割，语义分割，实例分割，目标检测等所有CV2d任务通用的即插即用模块
'''
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=2, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out,kernel_size,stride)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x
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
    input = torch.randn(3, 32, 64, 64)
    model = CMUNeXtBlock(ch_in=32,ch_out=64,kernel_size=3,stride=1)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
