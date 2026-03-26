import math

import torch
from einops import rearrange
from torch import nn
"""
AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters
Arxiv（2023）
即插即用模块：AKConv（可变形卷积核模块）
一、背景
卷积神经网络（CNNs）在深度学习领域成果显著，但标准卷积操作存在固有缺陷。其采样形状固定，局限于局部窗口，
无法获取其他位置信息；卷积核尺寸固定为 k×k，参数数量随尺寸呈平方增长，难以适应不同数据集中目标形状和大
小的变化。可变形卷积（Deformable Conv）虽通过偏移调整采样形状提升性能，但仍存在灵活性不足、参数增长方
式不友好等问题。AKConv 旨在解决这些问题，为卷积操作提供更多选择，平衡网络开销与性能。

二、AKConv 原理
1. 整体架构设计
A. 定义初始采样位置：为实现不规则卷积核的采样，提出一种算法生成任意大小卷积核的初始采样坐标。先按规则采样网
格生成部分坐标，再为剩余采样点创建不规则网格，最后拼接形成整体采样网格。以左上角 (0, 0) 点为采样原点，定义
相应卷积操作。
B. 可变形卷积操作：标准卷积采样位置固定，可变形卷积虽有改进但仍有局限。AKConv 通过卷积操作获取偏移，与原始
坐标相加得到修改后的坐标，再经插值和重采样获取对应位置特征。可通过多种方式提取不规则采样形状的特征，如按行或
列堆叠特征后用相应卷积操作，或变换特征维度后用特定卷积操作。
C. 扩展 AKConv：AKConv 设计新颖，即使不使用可变形卷积中的偏移思想，也能实现多种卷积核形状。可根据不同任务
设计相应形状的卷积操作，还可添加可学习偏移动态适应目标变化。

2. 卷积调制块核心组件
A. 坐标生成算法：通过特定算法生成卷积核的初始采样坐标，使不规则卷积核有合适的采样网格，为后续卷积操作提供基础。
B. 偏移调整机制：获取偏移以调整不规则卷积核的采样位置，适应目标的不同变化，实现更灵活的特征提取。
C. 特征提取方式：针对不规则采样形状，采用多种特征堆叠和卷积操作方式，确保有效提取特征。

3. 微观设计考量
A. 卷积核参数与形状灵活性：AKConv 允许卷积核有任意数量的参数和采样形状，突破了传统卷积和可变形卷积的限制，
为网络提供更丰富的选择。
B. 参数增长特性：卷积核参数数量呈线性增长，相较于传统卷积的平方增长，对硬件环境更友好，在减少模型参数和计
算开销方面有优势。
C. 初始采样形状设计：不同初始采样形状对网络性能有影响，针对特定网络和数据集，选择合适的初始采样形状有助于
提升网络性能。

三、适用任务
1. AKConv 适用于目标检测任务，在 COCO2017、VOC 7+12 和 VisDrone-DET2021 等代表性数据集上进行的实验
表明，它能有效提升网络的检测精度，可作为即插即用的卷积操作替代标准卷积，提高网络性能。
2. 适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
"""

class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
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

    # generating the inital sampled shapes for the AKConv with different sizes.
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
        # using Conv3d x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),
        # stride=(num_param,1,1),bias= False) using 1 × 1 Conv x_offset = x_offset.permute(0,1,4,2,3), then,
        # x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1),
        # bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = AKConv(inc=64,outc=64, num_param=3).cuda()
    input = torch.rand(3, 64, 56, 56).cuda()
    output = block(input)
    print(input.size(), output.size())
