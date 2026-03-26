import math
import torch
import torch.nn as nn
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')
# 论文：https://www.sciencedirect.com/science/article/abs/pii/S0262885624002956   SCI 2024
'''
LDConv：用于改进卷积神经网络的线性可变形卷积        SCI 2024
缺点：原始模块在处理卷积采样时存在一个显著缺点：采样密度是固定的，无论特征图的复杂性如何，采样数量都保持不变。
这种方式导致计算资源浪费，特别是在处理简单特征时不必要地增加了计算开销。同时，固定采样无法根据特征图的变化灵活调整密度，
难以在不同复杂度的图像区域中高效提取特征。

创新：为了解决这一问题，引入了动态稠密采样机制。该机制通过注意力机制计算出特征图的稠密度，根据稠密度自适应地调整采样点的数量。
当特征图区域较复杂时，稠密度较高，采样点数相应增多，从而更加精细地捕捉细节信息；而在简单区域，采样点数减少，从而节省计算资源。
具体来说，注意力机制在特征图上应用1×1卷积，通过输出的稠密度系数来动态控制采样点数，使采样点数量和特征图复杂度动态匹配，
实现计算效率和特征提取效果的平衡。
'''
import torch
import torch.nn as nn
from einops import rearrange
import math

import torch
import torch.nn as nn
from einops import rearrange
import math


class LDConv_DynamicSampling(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv_DynamicSampling, self).__init__()
        self.num_param = num_param
        self.stride = stride

        # 主卷积层，包含BN和激活函数，输出尺寸与输入一致
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )

        # 注意力机制模块，用于自适应调整采样密度
        self.attention = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 偏移卷积层，用于生成采样位置的偏移
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)

    def forward(self, x):
        # 生成自适应采样密度
        density = self.attention(x)
        adaptive_num_param = max(1, int(self.num_param * density.mean().item()))

        # 动态调整 num_param 保持与注意力权重一致
        if adaptive_num_param != self.num_param:
            self.num_param = adaptive_num_param
            self.p_conv = nn.Conv2d(x.size(1), 2 * self.num_param, kernel_size=3, padding=1, stride=self.stride)
            nn.init.constant_(self.p_conv.weight, 0)

        # 生成偏移
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2

        # 基于偏移生成采样位置
        p = self._get_p(offset, dtype)
        p = p.permute(0, 2, 3, 1).contiguous()

        # 计算采样点的四个邻近整数点 (q_lt, q_rb, q_lb, q_rt)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # 双线性插值权重计算
        g_lt = (1 + (q_lt[..., :N] - p[..., :N])) * (1 + (q_lt[..., N:] - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N] - p[..., :N])) * (1 - (q_rb[..., N:] - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N] - p[..., :N])) * (1 - (q_lb[..., N:] - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N] - p[..., :N])) * (1 + (q_rt[..., N:] - p[..., N:]))

        # 获取特征值并结合插值权重
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 结合四个邻近点的特征值，根据双线性插值计算最终特征 x_offset
        x_offset = (g_lt.unsqueeze(dim=1) * x_q_lt +
                    g_rb.unsqueeze(dim=1) * x_q_rb +
                    g_lb.unsqueeze(dim=1) * x_q_lb +
                    g_rt.unsqueeze(dim=1) * x_q_rt)

        # 调整 x_offset 形状以匹配卷积输入
        x_offset = self._reshape_x_offset(x_offset, self.num_param)

        # 使用卷积层得到最终输出
        out = self.conv(x_offset)
        return out

    def _get_p(self, offset, dtype):
        # 根据偏移生成采样位置
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 从输入特征 x 中获取偏移位置 q 处的特征值
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        # 将 index 限制在特征图的范围内
        index = (q[..., :N] * padded_w + q[..., N:]).long()
        index = torch.clamp(index, 0, x.size(-1) - 1)

        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        # 调整 x_offset 的形状以适应卷积输入
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

    def _get_p_n(self, N, dtype):
        # 初始化卷积核采样坐标
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        p_n_x, p_n_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int))
        p_n_x, p_n_y = torch.flatten(p_n_x), torch.flatten(p_n_y)
        p_n = torch.cat([p_n_x, p_n_y], 0).view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # 生成偏移为零的初始采样位置
        p_0_x, p_0_y = torch.meshgrid(torch.arange(0, h * self.stride, self.stride),
                                      torch.arange(0, w * self.stride, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0


if __name__ == '__main__':
    input = torch.rand(1, 32, 256, 256) #输入 B C H W,

    #LDConv_2D   # 输入 B C H W,  输出 B C H W
    model = LDConv_DynamicSampling(inc=32,outc=32,num_param=3)

    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())
