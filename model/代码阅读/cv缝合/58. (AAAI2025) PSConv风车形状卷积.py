import torch
import torch.nn as nn
import torch.nn.functional as F
'''
58. Pinwheel-shaped Convolution and Scale-based Dynamic Loss for 
Infrared Small Target Detection （AAAI 2025顶会）
即插即用模块：PSConv（风车形状卷积模块） 替身模块
一、背景
在红外小目标检测领域，基于卷积神经网络（CNN）的方法成果显著。然而，这些方法普遍采用的标准卷积存在缺陷，
它没有考虑到红外小目标像素分布的空间特征。与此同时，当前的损失函数在处理不同尺度目标时，没有充分考虑到
尺度和位置损失的敏感性差异，进而限制了对暗小目标的检测性能。另外，现有的真实拍摄的红外小目标检测数据集
也不理想，存在小目标比例低、背景简单、数据规模小等问题，无法满足复杂场景下的检测需求。

二、PSConv 模块原理
1. 整体结构
a. 输入特征：以特定的输入张量作为基础开启卷积操作流程。
b. 不对称卷积核与填充：PSConv 采用独特的不对称填充方式，制作出水平和垂直方向不同的卷积核，
以此来提取图像不同区域的特征。
c. 多卷积层操作：先进行第一层的并行卷积，然后把这些卷积结果拼接起来，最后通过归一化卷积对
输出特征图的尺寸进行调整。
d. 输出特征：最终得到的输出特征图可以和标准卷积层相互替换，并且还能像通道注意力机制一样，
分析不同卷积方向在特征提取中的贡献。
二、关键模块
a. 分组卷积：通过分组卷积技术，在扩大感受野的同时，还能减少所需的参数数量。
b. 批归一化（BN）和 SiLU 激活函数：每次卷积之后，都会使用批归一化（BN）和 SiLU 激活函数，
这样可以让训练过程更加稳定，速度也更快。
c. 计算复杂度：和标准卷积对比，PSConv 在扩大感受野的同时，增加的参数很少。也就是说，PSConv
能以很小的参数代价，有效扩大感受野。

三、适用于：红外小目标检测，小目标检测任务，目标检测，图像分割，语义分割，图像增强等所有一切计算
机视觉CV任务通用的即插即用卷积模块。
'''

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PSConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # 定义4种非对称填充方式，用于风车形状卷积的实现
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 每个元组表示 (左, 上, 右, 下) 填充
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建4个填充层

        # 定义水平方向卷积操作，卷积核大小为 (1, k)，步幅为 s，输出通道数为 c2 // 4
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

        # 定义垂直方向卷积操作，卷积核大小为 (k, 1)，步幅为 s，输出通道数为 c2 // 4
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # 最终合并卷积结果的卷积层，卷积核大小为 (2, 2)，输出通道数为 c2
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

        # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理，最终输出
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))  # 在通道维度拼接，并通过 cat 卷积层处理


# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module =  PSConv(c1=128,c2=128,k=3,s=1)
    input_tensor = torch.randn(1, 128, 128, 128)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状
