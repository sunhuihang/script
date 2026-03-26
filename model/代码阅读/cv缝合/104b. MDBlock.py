import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.branch5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.branch7x7 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, x):
        return self.fuse(torch.cat([
            self.branch3x3(x),
            self.branch5x5(x),
            self.branch7x7(x)
        ], dim=1))

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        ctx.eps = eps
        return weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        y, var, weight = ctx.saved_tensors
        eps = ctx.eps
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(1, keepdim=True)
        mean_gy = (g * y).mean(1, keepdim=True)
        gx = (1. / torch.sqrt(var + eps)) * (g - y * mean_gy - mean_g)
        return gx, grad_output.sum([0, 2, 3]), grad_output.sum([0, 2, 3]), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class DBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, extra_depth_wise=True):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.conv1 = nn.Conv2d(c, self.dw_channel, 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, 3, padding=1, groups=c) if extra_depth_wise else nn.Identity()

        self.msconv = MultiScaleConv(self.dw_channel)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, 1)
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, 1)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp, adapter=None):
        y = inp
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = self.msconv(x)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma

# ✅ 测试入口
if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)
    model = DBlock(c=64)
    output = model(input)
    print('MBlock 输入尺寸:', input.shape)
    print('MDBlock 输出尺寸:', output.shape)
