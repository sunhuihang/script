import torch
import torch.nn as nn
from timm.models.layers import DropPath

'''
Rewrite the Stars (CVPR2024)
背景：作者提出了一种名为“星操作”（Star Operation）的新方法，旨在利用逐元素乘法实现高维度、非线性特征空间的映射。
这种操作不同于传统网络通过增加网络宽度来提升维度的做法，能够在保持网络结构紧凑的情况下实现出色的性能。

创新点：
论文提出了“星操作”的概念，这是一种逐元素乘法，用于将不同子空间的特征融合起来。在自注意力机制（如Transformer）
占主导地位的背景下，星操作被证明可以更有效地聚合信息，特别是在特征维度较少时具有更高的效率。研究还发现，与传统的加法聚
合方式相比，星操作在图像分类等任务上表现更佳。

网络架构：
StarNet是一个多阶段的分层架构，共包含四个阶段（stage1到stage4）。图中从左到右展示了每个阶段的流程，每个阶段包
括卷积层和“星操作”模块（Star Blocks）。整个模型输入为图像，经过处理后最终输出给全局平均池化（GAP）和全连接层（FC）得到
分类结果。

Star模块细节：
1. DW-Conv：深度卷积（Depth-wise Convolution），卷积核大小为7，步长为1，主要用于空间信息的提取。
2. BN/ReLU6：批归一化（BN）和ReLU6激活函数，用于正则化和引入非线性。
3. FC：全连接层，通常用于通道间的信息混合和变换。
星操作：核心部分是星操作，即逐元素乘法，用于融合来自不同分支的特征。这种操作在不显著增加计算量的情况下，可以将特征映射到
高维空间，增强表达能力。

'''
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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


class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block =Star_Block(32)
    input = torch.rand(1, 32, 64, 64)
    output = block(input)
    print("input.shape:", input.shape)
    print("output.shape:",output.shape)