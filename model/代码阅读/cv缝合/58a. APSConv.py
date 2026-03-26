import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新：Attention Pinwheel  Convolution (APSConv)
改进说明：
1. 新增ChannelAttention类：定义了通道注意力机制模块，通过全局平均池化和全局最大池化获取通道特征，
然后经过两层全连接层（这里用1x1卷积实现），最后使用fesigmoid激活函数得到通道注意力权重，再与原始特
征图相乘，突出重要通道。
2. 修改PSConv类：在PSConv类的__init__方法中添加了self.channel_att = ChannelAttention(c2)，
初始化通道注意力模块；在forward方法中，对经过cat卷积层处理后的特征图应用通道注意力机制，即return
 self.channel_att(cat_result) 。
"""

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


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


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

        self.channel_att = ChannelAttention(c2)

    def forward(self, x):
        # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

        # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理
        cat_result = self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

        # 应用通道注意力机制
        return self.channel_att(cat_result)


# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module = PSConv(c1=128, c2=128, k=3, s=1)
    input_tensor = torch.randn(1, 128, 128, 128)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状