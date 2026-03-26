import torch
import torch.nn as nn
from torch import Tensor

class HalfConv(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)                                                                              # 微信公众号:CV缝合救星

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)                                                                                               # 微信公众号:CV缝合救星
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

class CGHalfConv(nn.Module):
    def __init__(self, dim):
        super(CGHalfConv, self).__init__()
        self.div_dim = int(dim / 3)
        self.remainder_dim = dim % 3
        self.p1 = HalfConv(self.div_dim, 2)
        self.p2 = HalfConv(self.div_dim, 2)
        self.p3 = HalfConv(self.div_dim + self.remainder_dim, 2)                                                                                               # 微信公众号:CV缝合救星

    def forward(self, x):
        y = x
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.div_dim + self.remainder_dim], dim=1)                                                                                               # 微信公众号:CV缝合救星
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        x = torch.cat((x1, x2, x3), 1)
        return x + y
    
if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 256, 256)

    # 初始化
    cghalfconv = CGHalfConv(dim=32)

    # 前向传播测试
    output = cghalfconv(x)

    # 输出结果形状
    print(cghalfconv)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]                                                                                               # 微信公众号:CV缝合救星
    print("输出张量形状:", output.shape)  # [B, C, H, W]                                                                                                   # 微信公众号:CV缝合救星