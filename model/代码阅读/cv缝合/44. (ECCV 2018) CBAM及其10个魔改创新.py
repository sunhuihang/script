import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 通道注意力的结构：全局平均池化 + 全局最大池化 + MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        avg_out = self.avg_pool(x)  # (batch_size, channels, 1, 1)
        max_out = self.max_pool(x)  # (batch_size, channels, 1, 1)

        # 通过两个池化分支生成特征表示
        avg_out = self.fc2(F.relu(self.fc1(avg_out)))  # (batch_size, channels, 1, 1)
        max_out = self.fc2(F.relu(self.fc1(max_out)))  # (batch_size, channels, 1, 1)

        # 通道注意力通过sigmoid激活
        out = avg_out + max_out  # (batch_size, channels, 1, 1)
        return self.sigmoid(out)  # (batch_size, channels, 1, 1)


# 定义空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 空间注意力模块：使用7x7卷积
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)

        # 将平均池化和最大池化结果沿通道维拼接
        out = torch.cat([avg_out, max_out], dim=1)  # (batch_size, 2, height, width)

        # 使用7x7卷积生成空间注意力图
        attention = self.conv(out)  # (batch_size, 1, height, width)

        # 激活后得到最终的空间注意力图
        return self.sigmoid(attention)  # (batch_size, 1, height, width)


# 定义CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 初始化通道注意力和空间注意力模块
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 先通过通道注意力
        x = x * self.channel_attention(x)  # (batch_size, channels, height, width)

        # 再通过空间注意力
        x = x * self.spatial_attention(x)  # (batch_size, channels, height, width)

        return x


# 测试代码
if __name__ == "__main__":
    # 假设输入特征图的大小为 (batch_size=2, channels=64, height=32, width=32)
    input_tensor = torch.randn(2, 64, 32, 32)

    # 创建CBAM模块
    cbam = CBAM(in_channels=64)

    # 通过CBAM模块
    output_tensor = cbam(input_tensor)

    print("Output shape:", output_tensor.shape)
