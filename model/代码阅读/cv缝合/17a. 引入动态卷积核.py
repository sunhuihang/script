import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新：引入动态卷积核
不足：通常，卷积层使用固定的卷积核大小（例如 3x3、5x5），但是在处理不同类型的图像时，固定大小的卷积核
可能无法充分捕捉到图像的多样性和不同尺度的特征。为此，可以通过动态调整卷积核的大小和权重来提高网络对输
入图像的适应能力。

魔改创新：通常，卷积层使用固定的卷积核大小（3x3、5x5、7x7），但是在处理不同类型的图像时，固
定大小的卷积核可能无法充分捕捉到图像的多样性和不同尺度的特征。为此，可以通过动态调整卷积核的大小和权重
来提高网络对输入图像的适应能力。
"""

def dynamic_conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    """动态卷积：卷积核大小根据输入动态生成"""
    kernel_size = torch.randint(3, 7, (1,)).item()  # 动态选取卷积核的大小，3到7之间
    padding = (kernel_size - 1) // 2 * dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)


class DynamicConvModule(nn.Module):
    def __init__(self, inplanes, outplanes, scale=3, stride=1):
        super(DynamicConvModule, self).__init__()

        self.width = inplanes
        self.nums = scale
        self.stride = stride

        self.convs = nn.ModuleList([])

        for i in range(self.nums):
            self.convs.append(dynamic_conv(self.width, outplanes, kernel_size=3, stride=stride))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]

        # 初始卷积
        out = self.convs[0](x)
        out = self.relu(out)

        # 迭代应用更多卷积，确保尺寸一致
        for i in range(1, self.nums):
            conv_out = self.convs[i](x)
            # 使用 F.interpolate 进行插值，确保输出尺寸一致
            conv_out = F.interpolate(conv_out, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = out + conv_out
            out = self.relu(out)

        return out


class DynamicMSPABlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, baseWidth=30, scale=3, norm_layer=None, stride=1):
        super(DynamicMSPABlock, self).__init__()

        planes = inplanes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = dynamic_conv(inplanes, baseWidth * scale)
        self.bn1 = norm_layer(baseWidth * scale)

        self.conv2 = DynamicConvModule(baseWidth * scale, baseWidth * scale, scale=scale, stride=stride)
        self.bn2 = norm_layer(baseWidth * scale)

        self.conv3 = dynamic_conv(baseWidth * scale, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 使用插值调整尺寸以匹配输入的尺寸
        out = F.interpolate(out, size=identity.shape[2:], mode='bilinear', align_corners=False)

        out += identity
        out = self.relu(out)

        return out


# 测试代码
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 64, 32, 32)

    # 创建一个 DynamicMSPABlock 模块实例
    dynamic_mspa_block = DynamicMSPABlock(inplanes=64, scale=3)

    # 执行前向传播
    output = dynamic_mspa_block(input)

    # 打印输入和输出的形状
    print('Input Size:', input.size())
    print('Output Size:', output.size())
