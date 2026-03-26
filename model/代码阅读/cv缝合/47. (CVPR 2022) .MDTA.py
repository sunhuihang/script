import torch
import torch.nn as nn
import math
from einops import rearrange
"""
Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)
一、背景
在图像恢复领域Transformer中的自注意力机制计算复杂度高，难以应用于高分辨率图像，而CNN又存在固有缺陷。
为克服这些问题，MDTA 模块被提出，旨在实现高效的特征处理与长程像素关系建模。

二、MDTA 原理详细解读
1. 局部上下文融合与投影生成：先对层归一化后的张量进行处理。用 1×1 卷积聚合逐像素跨通道上下文，再用
3×3 深度卷积编码通道空间上下文，从而生成查询、键和值投影，使它们富含局部上下文信息。
2. 跨通道注意力构建：重塑查询和键投影，让它们的点积交互产生大小为的转置注意力图。其计算过程是先通过特
定公式计算注意力权重，再将其应用于值投影来更新特征。并且像传统多头注意力那样，将通道分组为 “头” 并行学
习注意力图。

三、适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
"""
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = Attention(64).cuda()
    input = torch.rand(3, 64, 85, 85).cuda()
    output = block(input)
    print(input.size(), output.size())
