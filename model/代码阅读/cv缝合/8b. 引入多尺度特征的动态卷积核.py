import torch
import torch.nn as nn
import torch.nn.functional as F
"""
局限性：虽然大卷积核能够一定程度上提取全局信息，但在高度复杂的医学图像中，
依赖大卷积核可能仍不足以捕捉更深层次的全局特征。例如，对于形态多样或位置随机的病灶，
卷积操作可能无法完全捕获长距离的依赖关系。

思路：设计一个动态卷积核模块，根据输入图像的特征自适应调整卷积核大小，从而增强对不同尺度病灶的适应性。
这种方法可以解决不同病灶尺度不一致的问题。

实现：将CMUNeXt模块中的大卷积核替换为一个多尺度卷积核模块，根据不同的特征层生成不同大小的卷积核，
在特征提取过程中动态生成3x3、5x5、7x7等卷积核进行融合，确保网络在不同尺度下都有较强的特征提取能力。
"""


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], reduction=4):
        super(MultiScaleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.reduction = reduction

        # Define depthwise convolutions with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=1, padding=k // 2, groups=in_channels)
            for k in kernel_sizes
        ])

        # Pointwise convolution to combine channels
        self.pointwise = nn.Conv2d(len(kernel_sizes) * in_channels, out_channels, kernel_size=1)

        # Inverted bottleneck design to expand and then reduce channels
        self.expand = nn.Conv2d(out_channels, out_channels * reduction, kernel_size=1)
        self.reduce = nn.Conv2d(out_channels * reduction, out_channels, kernel_size=1)

        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Apply each depthwise convolution and concatenate results along channel dimension
        multi_scale_features = [conv(x) for conv in self.convs]
        x = torch.cat(multi_scale_features, dim=1)

        # Apply pointwise convolution to mix channels
        x = self.pointwise(x)

        # Inverted bottleneck: expand and then reduce channels
        x = self.gelu(self.expand(x))
        x = self.reduce(x)

        # Apply batch normalization and GELU activation
        x = self.bn(x)
        x = self.gelu(x)

        return x


# Test the MultiScaleConvBlock
if __name__ == "__main__":
    # Example input tensor (batch_size=1, channels=32, height=64, width=64)
    input_tensor = torch.randn(3, 32, 64, 64)
    multi_scale_conv_block = MultiScaleConvBlock(in_channels=32, out_channels=32)
    output = multi_scale_conv_block(input_tensor)
    print('input_size:', input_tensor.shape)
    print("Output shape:", output.shape)
