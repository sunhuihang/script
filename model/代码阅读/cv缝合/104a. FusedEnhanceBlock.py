import torch
import torch.nn as nn
import torch.nn.functional as F

# 简单门控结构
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

# 频域 MLP 模块
class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process_mag = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_mag(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

# 魔改模块：FusedEnhanceBlock
class FusedEnhanceBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1, 4, 9]):
        super(FusedEnhanceBlock, self).__init__()
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1)
        self.branches = nn.ModuleList([
            nn.Conv2d(self.dw_channel, self.dw_channel, 3, padding=d, dilation=d, groups=self.dw_channel)
            for d in dilations
        ])

        self.sg = SimpleGate()
        self.ca = ChannelAttention(self.dw_channel // 2)
        self.conv2 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1)

        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)

        self.freq = FreMLP(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        y = x
        x = self.norm1(x)
        x = self.conv1(x)
        z = sum([branch(x) for branch in self.branches])
        z = self.sg(z)
        z = self.ca(z)
        z = self.conv2(z)
        y = y + self.beta * z

        x_freq = self.freq(self.norm2(y))
        out = y + self.gamma * (y * x_freq)
        return out

# 测试代码
if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    model = FusedEnhanceBlock(c=64)
    y = model(x)
    print(model)
    print('哔哩哔哩CV缝合救星-EBlock input_size:', x.size())
    print('哔哩哔哩CV缝合救星-EBlock output_size:', y.size())
