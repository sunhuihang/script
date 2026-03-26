import torch
import torch.nn as nn


# 定义瓶颈卷积模块
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels,
                                   bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


# 获取归一化层
def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm2d(channels)


# 定义GBConv模块
class SpatialGuidedConvolution(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(SpatialGuidedConvolution, self).__init__()

        # 自适应卷积核大小，增强对不同图像尺寸的适应性
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

        # 简单的几何先验引导模块
        self.sgc_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = x

        # 常规卷积块
        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)

        # 添加空间引导卷积（SGC）
        sgc_attention = self.sgc_attention(x1)
        x = x1 * x2 * sgc_attention
        x = self.block4(x)

        return x + residual


# 测试SpatialGuidedConvolution模块
if __name__ == "__main__":
    input = torch.randn(1, 32, 64, 64)  # 创建一个形状为 (1, 32, 64, 64) 的输入张量
    SGC = SpatialGuidedConvolution(32)
    print(SGC)
    output = SGC(input)  # 通过SpatialGuidedConvolution模块计算输出
    print("B站CV缝合救星：102a. SGC")
    print('Input size:', input.size())  # 打印输入张量的形状
    print('Output size:', output.size())  # 打印输出张量的形状
