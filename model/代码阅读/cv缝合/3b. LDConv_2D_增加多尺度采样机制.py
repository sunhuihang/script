import torch
import torch.nn as nn
from einops import rearrange
import math

"""
LDConv：用于改进卷积神经网络的线性可变形卷积        SCI 2024
增加多尺度采样机制
LDConv采样是基于单一尺度的，而在目标检测、语义分割等任务中，不同尺度的特征往往能更全面地捕获目标信息。
引入多尺度采样可以提升模型对多样化目标的适应性.

方法：在偏移生成阶段，为每个采样位置生成多个尺度的偏移，即在不同的空间尺度上进行采样。可以结合特征金字塔网络（FPN）或使用多层采样，
以低、中、高三个尺度采样后融合特征。
优势：多尺度采样可以增强模型对不同尺寸和复杂度目标的感知能力，尤其是在多物体检测或分割任务中，能够提高模型的适应性和性能。
并且通过并行化采样处理，有望提升推理速度。
"""
import torch
import torch.nn as nn
from einops import rearrange
import math


class LDConv_MultiScale(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, scales=(1, 2, 3), bias=None):
        super(LDConv_MultiScale, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.scales = scales  # 定义多尺度采样的不同尺度

        # 为每个尺度定义卷积层
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=1, bias=bias),
                nn.BatchNorm2d(outc),
                nn.SiLU()
            ) for _ in scales
        ])

        # 为每个尺度定义偏移卷积层
        self.p_convs = nn.ModuleList([
            nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride) for _ in scales
        ])
        for p_conv in self.p_convs:
            nn.init.constant_(p_conv.weight, 0)

    def forward(self, x):
        scale_outputs = []

        for scale, p_conv, conv in zip(self.scales, self.p_convs, self.convs):
            # 为当前尺度生成偏移和采样位置
            offset = p_conv(x) * scale  # 调整偏移幅度以匹配当前尺度
            dtype = offset.data.type()
            N = offset.size(1) // 2

            # 多尺度偏移调整
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

            # 在当前尺度上进行卷积
            scale_outputs.append(conv(x_offset))

        # 将所有尺度的输出进行融合，确保输出的空间尺寸与输入相同
        out = torch.mean(torch.stack(scale_outputs, dim=0), dim=0)  # 取平均保留输入的空间维度
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
        # 沿着最后一个采样维度 n 取平均，确保 h 和 w 维度保持不变
        x_offset = x_offset.mean(dim=-1)
        return x_offset

    def _get_p_n(self, N, dtype):
        # 初始化卷积核采样坐标
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        p_n_x, p_n_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int), indexing='ij')
        p_n_x, p_n_y = torch.flatten(p_n_x), torch.flatten(p_n_y)

        # 确保 p_n_x 和 p_n_y 的元素数量为 N
        if p_n_x.numel() < N:
            p_n_x = torch.cat([p_n_x, p_n_x[:N - p_n_x.numel()]])
            p_n_y = torch.cat([p_n_y, p_n_y[:N - p_n_y.numel()]])
        p_n_x, p_n_y = p_n_x[:N], p_n_y[:N]

        # 拼接为 p_n
        p_n = torch.cat([p_n_x, p_n_y], 0).view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # 生成偏移为零的初始采样位置
        p_0_x, p_0_y = torch.meshgrid(torch.arange(0, h * self.stride, self.stride),
                                      torch.arange(0, w * self.stride, self.stride), indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0


if __name__ == '__main__':
    input = torch.rand(1, 32, 256, 256) #输入 B C H W,

    #LDConv_2D   # 输入 B C H W,  输出 B C H W
    model = LDConv_MultiScale(inc=32,outc=32,num_param=3)

    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())