import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class MultiScaleLKP(nn.Module):
    """ 多尺度大核感知：生成动态卷积权重 """
    def __init__(self, dim, lks_list=[7, 11], sks=3, groups=8):
        super().__init__()
        self.paths = nn.ModuleList()
        for lks in lks_list:
            self.paths.append(
                nn.Sequential(
                    Conv2d_BN(dim, dim // 2),
                    nn.ReLU(),
                    Conv2d_BN(dim // 2, dim // 2, ks=lks,
                              pad=(lks - 1) // 2, groups=dim // 2),
                    Conv2d_BN(dim // 2, dim // 2),
                    nn.Conv2d(dim // 2, sks ** 2 * dim, kernel_size=1)
                )
            )
        self.norm = nn.GroupNorm(num_groups=dim // groups,
                                 num_channels=sks ** 2 * dim)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        outs = [p(x) for p in self.paths]  # 多尺度
        w = sum(outs) / len(outs)          # 融合
        w = self.norm(w)
        B, _, H, W = w.size()
        w = w.view(B, self.groups, self.dim // self.groups,
                   self.sks ** 2, H, W)
        return w


class SKA(nn.Module):
    """ 小核动态聚合 """
    def __init__(self, sks=3, groups=8):
        super().__init__()
        self.sks = sks
        self.groups = groups
        self.pad = (sks - 1) // 2

    def forward(self, x, w):
        B, C, H, W = x.shape
        g = self.groups
        Cg = C // g

        patches = F.unfold(x, kernel_size=self.sks, padding=self.pad)
        patches = patches.view(B, g, Cg, self.sks**2, H, W)

        out = (patches * w).sum(dim=3)  # [B,g,Cg,H,W]
        out = out.view(B, C, H, W)
        return out


class SEBlock(nn.Module):
    """ 通道注意力模块 """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class DCLSConv(nn.Module):
    """ Dual-Context Large-Small Convolution """
    def __init__(self, dim, lks_list=[7, 11], sks=3, groups=8, reduction=4):
        super().__init__()
        self.lkp = MultiScaleLKP(dim, lks_list=lks_list, sks=sks, groups=groups)
        self.ska = SKA(sks=sks, groups=groups)
        self.se = SEBlock(dim, reduction=reduction)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.ska(x, self.lkp(x))
        out = self.se(out)  # 通道增强
        return self.bn(out) + x


if __name__ == "__main__":
    # 配置输入张量
    batch_size = 1
    channels = 32
    height = 64
    width = 64

    x = torch.randn(batch_size, channels, height, width)

    model = DCLSConv(dim=channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)

    output = model(x)
    print(model)
    print("\n输入张量形状:", x.shape)
    print("\n微信公众号|Bilibili CV缝合救星")
    print("输出张量形状:", output.shape)
