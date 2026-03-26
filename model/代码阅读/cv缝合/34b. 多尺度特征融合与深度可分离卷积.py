import torch
import torch.nn as nn
import torch.nn.functional as F
"""
创新点：多尺度特征融合与深度可分离卷积
在医学图像分割中，不同尺度的特征通常包含了不同层次的空间信息。多尺度特征融合可以帮助模型捕捉更全面的细节信息。
此外，采用深度可分离卷积可以减少模型的参数数量和计算复杂度，同时保持较好的性能。
实现思路：
1. 多尺度融合：将输入的低层和高层特征在不同的尺度上进行融合。通过不同的步幅（stride）或者不同的卷积核尺寸
（kernel size），我们可以处理更广泛的空间信息。
2. 深度可分离卷积：使用深度可分离卷积代替常规卷积，来减少参数量和计算量。深度可分离卷积将传统卷积操作分解为两个步骤：
首先是深度卷积，每个通道单独卷积；然后是逐点卷积，对所有通道进行混合。
"""

# 深度可分离卷积模块
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 多尺度特征融合模块
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(MultiScaleFeatureFusion, self).__init__()

        self.input_dim = input_dim
        # 不同尺度的卷积操作
        self.conv1 = DepthwiseSeparableConv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConv2d(input_dim, output_dim, kernel_size=5, stride=1, padding=2)
        self.conv3 = DepthwiseSeparableConv2d(input_dim, output_dim, kernel_size=7, stride=1, padding=3)

        # 特征融合层，修改输入通道数为 384
        self.fusion_conv = nn.Conv2d(output_dim * 6, output_dim, kernel_size=1, bias=False)

    def forward(self, L_feature, H_feature):
        # 对低层和高层特征进行不同尺度卷积处理
        L_feature_1 = self.conv1(L_feature)
        L_feature_2 = self.conv2(L_feature)
        L_feature_3 = self.conv3(L_feature)
        H_feature_1 = self.conv1(H_feature)
        H_feature_2 = self.conv2(H_feature)
        H_feature_3 = self.conv3(H_feature)
        # 将不同尺度的特征融合
        fused_feature = torch.cat([L_feature_1, L_feature_2, L_feature_3,
                                   H_feature_1, H_feature_2, H_feature_3], dim=1)
        # 最终的卷积融合
        out = self.fusion_conv(fused_feature)
        return out

# 测试多尺度特征融合模块
if __name__ == '__main__':
    input1 = torch.randn(1, 64, 64, 64)  # 低层特征
    input2 = torch.randn(1, 64, 64, 64)  # 高层特征

    model = MultiScaleFeatureFusion(input_dim=64, output_dim=64)
    output = model(input1, input2)
    print("MultiScaleFeatureFusion Input size:", input1.size())
    print("MultiScaleFeatureFusion Output size:", output.size())
