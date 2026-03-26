import torch
import torch.nn as nn
from timm.models.layers import DropPath

'''
Star模块缺陷
星操作模块的缺点在于其特征表达不够灵活，高维特征的权重控制力不足，特别是在低通道数（网络宽度）条件下，可能无法充分发挥其高维特征映射的
优势。此外，由于逐元素乘法在不同硬件上的效率不同，可能导致性能不稳定。

CV缝合就行创新点：
引入自适应星操作（Adaptive Star Operation）模块，通过在星操作之前增加通道注意力机制，动态调整每个通道的权重，提升特征表达
的灵活性。同时，在逐元素乘法之后增加一个可学习的隐式高维权重矩阵，以控制每个隐式维度的特征权重。
'''
import torch
import torch.nn as nn
from timm.models.layers import DropPath

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SE_Block(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Adaptive_Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., reduction=4):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.se = SE_Block(dim, reduction)  # Add SE module for channel attention
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.hidden_weight = nn.Parameter(torch.ones(dim, 1, 1))  # learnable weight for high-dimensional control

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.se(x)  # Apply SE block
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = x * self.hidden_weight  # Apply learnable weight for implicit high-dimensional control
        x = input + self.drop_path(x)
        return x

# 测试代码
if __name__ == '__main__':
    block = Adaptive_Star_Block(32)
    input = torch.rand(1, 32, 64, 64)
    output = block(input)
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)
