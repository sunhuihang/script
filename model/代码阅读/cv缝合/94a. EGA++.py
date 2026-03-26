import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
import math


class MultiScaleGaussian(nn.Module):
    def __init__(self, dim, sizes, sigmas):
        super().__init__()
        self.filters = nn.ModuleList()
        for size, sigma in zip(sizes, sigmas):
            kernel = self.build_kernel(size, sigma)
            conv = nn.Conv2d(dim, dim, kernel_size=size, padding=size // 2, groups=dim, bias=False)
            conv.weight.data = kernel.repeat(dim, 1, 1, 1)
            conv.weight.requires_grad = False
            self.filters.append(conv)

    def forward(self, x):
        return sum(f(x) for f in self.filters) / len(self.filters)

    def build_kernel(self, size, sigma):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class ScharrEdge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        self.weight_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_x.weight.data = scharr_x.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.weight_y.weight.data = scharr_y.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.weight_x.weight.requires_grad = False
        self.weight_y.weight.requires_grad = False

    def forward(self, x):
        gx = self.weight_x(x)
        gy = self.weight_y(x)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return edge


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B,1,C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B,C,1,1)
        return x * y.expand_as(x)


class EGAPlusPlus(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.gaussian = MultiScaleGaussian(dim, sizes=[3, 5], sigmas=[0.8, 1.2])
        self.scharr = ScharrEdge(dim)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        self.eca = ECA(dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            build_norm_layer(norm_layer, dim)[1],
            act_layer()
        )

    def forward(self, x):
        g = self.gaussian(x)
        e = self.scharr(x)
        fuse = self.norm(self.act(x + g + e))
        out = self.eca(fuse)
        return self.fuse(out)


# 测试入口
if __name__ == "__main__":
    batch = 1
    channels = 32
    height, width = 128, 128
    x = torch.randn(batch, channels, height, width)

    ega_pp = EGAPlusPlus(dim=channels, norm_layer=dict(type='BN', requires_grad=True), act_layer=nn.ReLU)
    out = ega_pp(x)

    print("EGA++ 输入维度：", x.shape)
    print("\n哔哩哔哩: CV缝合救星!\n")
    print("EGA++ 输出维度：", out.shape)
