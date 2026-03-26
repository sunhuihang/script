import os
import sys
import inspect

from torch import nn
import torch
'''
Partial Convolution for Efficient Neural Networks (CVPR 2023)
即插即用模块：PConv（替身模块）

一、背景：
在边缘设备和移动设备上，神经网络的计算效率非常重要。传统卷积神经网络的计算量大，对硬件资源要求高。减少浮点运算次数（FLOPs）虽然能降低计算复杂度，
但推理速度并未明显提高。深度卷积和组卷积增加了内存访问频率，导致延迟问题。为解决这个问题，本文提出部分卷积（PConv）。部分卷积只对特定通道进行卷积，
减少冗余计算和内存访问。结合逐点卷积，PConv 保持了网络的特征提取能力。PConv在多种设备上展示了显著的性能提升，特别适合低计算资源的场景。

二、PConv模块机制：
1. 输入：特征图 X，包含高分辨率的空间信息。
2. 部分通道卷积
A. 输入特征图 X 的一部分通道用于卷积，而其余部分通道保持不变，降低了内存访问。
B. 卷积操作只作用于选中的部分通道，而这些通道足以代表整体的特征信息。
3. 特征融合
A. 在执行部分卷积后，通过逐点卷积（Pointwise Convolution, PWConv）将所有通道进行融合，以确保所有通道的信息均得到处理。
B. 逐点卷积将保留特征图中的全局信息并完成跨通道的特征整合。
4. 通道选择
A. 在进行通道融合之后，通过逐点卷积调整特征图的通道数，使网络结构更加灵活且能够适应不同的计算资源需求。

三、适用任务：目标检测，图像分割，分类任务等所有CV任务，尤其是需要在低延迟、低计算资源设备（如边缘计算、移动设备）上的应用。
'''



class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


if __name__ == '__main__':
    block = Partial_conv3(64, 2, 'split_cat').cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
