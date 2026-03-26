import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
        return x * channel_attention

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DGSAM(nn.Module):
    def __init__(self, channel, group=8, cov1=1, cov2=1):
        super().__init__()
        self.group = group
        self.cov1 = nn.Conv2d(channel, channel, kernel_size=1) if cov1 else None
        self.cov2 = nn.Conv2d(channel, channel, kernel_size=1) if cov2 else None

        # 初始化分组 CBAM
        self.cbam = nn.ModuleList([CBAM(channel // group) for _ in range(group)])

        # 残差动态缩放因子
        self.res_scale = nn.Parameter(torch.ones(1))

    def channel_shuffle(self, x, groups):
        B, C, H, W = x.size()
        x = x.view(B, groups, C // groups, H, W)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, x):
        x0 = x
        if self.cov1 is not None:
            x = self.cov1(x)

        # 分组
        y = torch.chunk(x, self.group, dim=1)
        out = []

        for y_, cbam in zip(y, self.cbam):
            y_cbam = cbam(y_)

            # 动态门控：Sigmoid + Tanh 融合
            gate_sigmoid = torch.sigmoid(y_cbam)
            gate_tanh = torch.tanh(y_cbam)
            gate = gate_sigmoid * gate_tanh

            # 均值门控重分配
            mean = gate.mean([1,2,3], keepdim=True)
            gate = torch.where(gate > mean, torch.ones_like(gate), gate)

            out.append(y_ * gate)

        # 拼接 + 通道打乱
        x = torch.cat(out, dim=1)
        x = self.channel_shuffle(x, self.group)

        if self.cov2 is not None:
            x = self.cov2(x)

        # Adaptive Residual Scaling
        x = x + self.res_scale * x0
        return x

if __name__ == '__main__':
    # 创建输入张量：形状 [B, C, H, W]
    x = torch.randn(1, 32, 256, 256).cuda()

    # 实例化
    model = DGSAM(32, group=8).cuda()
    output = model(x)

    # 打印输入输出形状
    print(model)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("哔哩哔哩CV缝合救星：🔥 DGSAM 运行成功！")
