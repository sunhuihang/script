import math
import torch
import torch.nn as nn
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')
# 论文：https://www.sciencedirect.com/science/article/abs/pii/S0262885624002956   SCI 2024
'''
LDConv：用于改进卷积神经网络的线性可变形卷积        SCI 2024
LDConv模块背景
卷积神经网络（CNN）在深度学习领域取得了广泛应用，从图像分类、物体检测到图像分割等任务都依赖于其卷积运算来提取局部特征。
然而，传统的卷积操作具有两个主要限制：
1.首先，卷积运算的采样位置是固定的局部窗口，无法有效捕获全局信息；其次，卷积核的大小是固定的，通常为正方形（如3×3或5×5），随着卷积核的增大，
参数数量呈平方增长。
2. 可变形卷积（Deformable Conv）虽然在一定程度上缓解了采样位置固定的问题，通过偏移调整来适应目标形状的变化，但其参数增长仍然呈平方趋势，
这对于硬件性能带来了较大挑战，尤其在处理大卷积核时会导致计算和内存开销大幅增加。

LDConv的局限性与创新
LDConv（线性可变形卷积）在以上背景下被提出，以应对传统卷积和可变形卷积的局限性。
与Deformable Conv不同，LDConv支持任意采样形状和任意数量的参数，可以根据任务需求灵活选择采样位置和卷积参数，且参数数量的增长呈线性趋势，
适用于更广泛的硬件环境和多样的任务需求。

LDConv的创新在于：
1. 灵活采样：支持非规则采样，使其能够适应不同目标的多变形状，提高卷积操作的特征提取能力。
2. 线性增长的参数：不同于标准和可变形卷积的平方增长，LDConv的参数数量随卷积核大小呈线性增长，显著降低了计算和内存开销。
3. 动态调整：引入偏移调整，允许每个位置的采样形状进行动态变化，以更好适应目标的多变特征。

LDConv的适用任务
LDConv作为即插即用的卷积模块，适用于几乎所有的计算机视觉任务，尤其在目标检测、图像分割和场景理解等任务中表现出色。
在COCO2017、VOC 7+12和VisDrone-DET2021等大型数据集上的实验结果表明，LDConv可以提升模型在不同物体大小上的检测精度，
并且在保证模型高效性的同时降低计算成本。

LDConv的设计还可以方便地集成到现有的模型模块中，例如将其应用于YOLO系列、FasterBlock和GSBottleneck等结构中，
通过更丰富的采样形状和灵活的参数控制来提升模型性能，使其在资源有限的硬件环境中仍能高效运行。
'''


class LDConv_2D(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv_2D, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


if __name__ == '__main__':
    input = torch.rand(1, 32, 256, 256) #输入 B C H W,
    #LDConv_2D   # 输入 B C H W,  输出 B C H W
    model = LDConv_2D(inc=32,outc=32,num_param=3)
    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())
