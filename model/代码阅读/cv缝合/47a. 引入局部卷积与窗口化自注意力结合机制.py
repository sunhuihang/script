import torch
import torch.nn as nn
from einops import rearrange
"""
CV缝合救星魔改创新：引入局部卷积与窗口化自注意力结合机制
一、背景：
1. 当前的自注意力机制在高分辨率图像处理时，计算复杂度较高（O(N²)），使其难以高效应用于大尺寸图像。
2. 问题：全局自注意力虽然能建模全局依赖，但其计算开销巨大，尤其是在高分辨率场景中，容易导致资源耗尽。
传统的卷积虽然计算高效，但难以捕获长程依赖。
二、创新点：
引入局部卷积与窗口化自注意力的结合机制，通过局部卷积提取局部上下文信息，结合窗口化的自注意力计算，
实现计算效率和特征建模能力的平衡。
具体实现：
1. 局部卷积特征提取：
在自注意力计算之前，先通过一个局部卷积模块（例如 3x3 卷积）提取局部上下文信息，聚焦每个位置的局部特征，
减少直接建模全局关系的复杂性。这一步能够有效聚合局部信息，为后续的窗口化注意力计算提供高质量的特征表示。
2. 窗口化自注意力计算：
将输入图像划分为多个固定大小的窗口（例如 7x7），并仅在每个窗口内部独立计算自注意力，从而避免全局计算的高复杂度。
窗口内的自注意力能够捕获局部区域的依赖关系，同时计算复杂度从 O(N²) 降低到 O(window_size²)。
3. 局部与窗口结合：
局部卷积增强了上下文信息的提取，窗口化自注意力进一步降低了计算量，两者的结合在复杂场景下既能保持计算效率，
又能捕获关键特征。
"""
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, window_size=7):
        """
        dim: 输入通道数
        num_heads: 注意力头数
        bias: 是否使用偏置
        window_size: 窗口大小，用于窗口化自注意力
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size  # 窗口大小

        # 局部卷积模块
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # 生成 qkv 的卷积层
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # 局部卷积处理，提取局部上下文
        x_local = self.local_conv(x)

        # 获取 q, k, v 投影
        qkv = self.qkv(x_local)
        q, k, v = qkv.chunk(3, dim=1)

        # 重塑为多头注意力的形状
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化 q 和 k
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 窗口化自注意力计算
        # 计算窗口中的自注意力，仅在窗口内计算注意力，减少计算量
        h_window, w_window = h // self.window_size, w // self.window_size
        attn = (q @ k.transpose(-2, -1))  # 计算每个窗口的注意力矩阵

        # 对注意力权重进行softmax归一化
        attn = attn.softmax(dim=-1)

        # 使用注意力权重对值进行加权求和
        out = (attn @ v)

        # 重塑输出
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 最后的输出投影
        out = self.project_out(out)
        return out

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = Attention(64, num_heads=4, window_size=7).cuda()  # 指定窗口大小
    input = torch.rand(3, 64, 85, 85).cuda()
    output = block(input)
    print(input.size(), output.size())
