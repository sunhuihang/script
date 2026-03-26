import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math


# 哔哩哔哩：CV缝合救星
# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)  # B 6C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


# 多尺度特征提取模块
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, dim):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        return x1 + x2 + x3


# 先空间校正再通道校正
class MSFCM(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(MSFCM, self).__init__()
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.multi_scale_extractor = MultiScaleFeatureExtractor(dim)

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        # 提取多尺度特征
        x1_multi_scale = self.multi_scale_extractor(x1_1)
        x2_multi_scale = self.multi_scale_extractor(x2_1)

        channel_weights = self.channel_weights(x1_multi_scale, x2_multi_scale)

        main_out = x1_multi_scale + fuse_weights[1] * channel_weights[1] * x2_multi_scale
        aux_out = x2_multi_scale + fuse_weights[1] * channel_weights[0] * x1_multi_scale
        return main_out, aux_out


if __name__ == "__main__":
    import torch

    # 设置输入张量大小
    batch_size = 1
    channels = 32
    height, width = 128, 128  # 假设输入图像尺寸为128x128

    # 创建两个输入张量
    x1 = torch.randn(batch_size, channels, height, width).cuda()  # 输入张量1
    x2 = torch.randn(batch_size, channels, height, width).cuda()  # 输入张量2

    # 初始化 MSFCM 模块
    dim = channels
    msfcm = MSFCM(dim=dim, reduction=1).cuda()
    print(msfcm)
    print("\n哔哩哔哩：CV缝合救星！\n")

    # 前向传播测试
    main_out, aux_out = msfcm(x1, x2)

    # 打印输入和输出的形状
    print(f"Input shape (x1):           {x1.shape}")
    print(f"Input shape (x2):           {x2.shape}")
    print(f"Output shape (main_out):    {main_out.shape}")
    print(f"Output shape (aux_out):     {aux_out.shape}")
