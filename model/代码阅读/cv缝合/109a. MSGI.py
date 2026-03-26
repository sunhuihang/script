import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# CV缝合救星2025.06.24视频
# 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sa = self.sigmoid(self.conv(x))
        return x * sa

# CV缝合救星2025.06.24视频
# 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
class MSGI_FCM(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        # 多尺度分支
        self.conv1 = Conv(dim, dim // 2, k=1)
        self.conv3 = Conv(dim, dim // 2, k=3)
        self.conv5 = Conv(dim, dim // 2, k=5)
        self.merge = Conv(dim // 2 * 3, dim, k=1)

        # 注意力模块
        self.channel_attn = ChannelAttention(dim)
        self.spatial_attn = SpatialAttention(dim)

        # 输出整合
        self.out_conv = Conv(dim, dim_out, k=1)
        self.residual = nn.Identity() if dim == dim_out else Conv(dim, dim_out, k=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)

        multi_scale = torch.cat([x1, x3, x5], dim=1)
        x_mixed = self.merge(multi_scale)

        x_c = self.channel_attn(x_mixed)
        x_s = self.spatial_attn(x_mixed)

        out = self.out_conv(x_c + x_s)
        return out + self.residual(x)


if __name__ == "__main__":
    # 测试入口
    x = torch.randn(1, 32, 256, 256)
    model = MSGI_FCM(dim=32, dim_out=32)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(model)
    print("哔哩哔哩：CV缝合救星！")
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
