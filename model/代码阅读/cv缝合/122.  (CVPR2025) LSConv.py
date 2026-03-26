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


class LKP(nn.Module):
    """ 大核感知：生成动态卷积权重 """
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks,
                             pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        # 修正这里：输出通道应该是 sks^2 * dim
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups,
                                 num_channels=sks ** 2 * dim)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))  # [B, sks^2 * C, H, W]
        B, _, H, W = w.size()
        w = w.view(B, self.groups, self.dim // self.groups,
                   self.sks ** 2, H, W)
        return w



class SKA(nn.Module):
    """ 小核动态聚合（纯 PyTorch 实现，代替 Triton 内核） """
    def __init__(self, sks=3, groups=8):
        super().__init__()
        self.sks = sks
        self.groups = groups
        self.pad = (sks - 1) // 2

    def forward(self, x, w):
        B, C, H, W = x.shape
        g = self.groups
        Cg = C // g

        # unfold: [B, C*sks^2, H*W]
        patches = F.unfold(x, kernel_size=self.sks, padding=self.pad)
        patches = patches.view(B, g, Cg, self.sks**2, H, W)  # [B,g,Cg,K^2,H,W]

        # 权重 reshape: [B,g,Cg,K^2,H,W]
        w = w.view(B, g, Cg, self.sks**2, H, W)

        # 动态加权求和
        out = (patches * w).sum(dim=3)  # [B,g,Cg,H,W]
        out = out.view(B, C, H, W)
        return out



class LSConv(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        self.lkp = LKP(dim, lks=lks, sks=sks, groups=groups)
        self.ska = SKA(sks=sks, groups=groups)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


if __name__ == "__main__":
    # 配置输入张量
    batch_size = 1
    channels = 32
    height = 64
    width = 64

    x = torch.randn(batch_size, channels, height, width)

    model = LSConv(dim=channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)

    output = model(x)
    print(model)
    print("\n输入张量形状:", x.shape)
    print("\n微信公众号|Bilibli CV缝合救星")
    print("输出张量形状:", output.shape)
