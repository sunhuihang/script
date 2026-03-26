import torch
import torch.nn as nn
from typing import Tuple, Union

from ultralytics.nn.modules import C3
"""
CV缝合救星魔改创新：引入动态空间自适应机制
一、背景：
1. 在当前的 MaSA 模块中，虽然基于曼哈顿距离的空间衰减矩阵和注意力分解形式已经在一定程度上提升了模型性能，
但可以进一步改进。创新点在于为 MaSA 增加一个动态空间自适应模块，使其能够根据输入图像的内容和特征分布动态
调整空间衰减的权重和范围。
2. 具体实现方式如下：在 MaSA 模块的前向传播过程中，首先对输入的特征图进行全局池化操作，得到一个低维的特
征向量表示。然后通过一个小型的神经网络（例如由几个全连接层组成）对这个特征向量进行处理，输出一组动态调整
参数。这些参数将用于实时修改空间衰减矩阵中的衰减系数以及注意力分解过程中的相关权重。对于空间衰减矩阵，动态
参数可以根据图像中不同区域的纹理复杂度、目标大小等因素，在不同位置上更加灵活地控制衰减程度。在注意力分解方
面，动态参数可以根据特征的重要性分布，自适应地调整水平和垂直方向上的注意力分配权重。
二、这样的创新点具有以下优势：
1. 能够更好地适应不同类型和场景的图像数据，提高模型在复杂多变的视觉任务中的泛化能力。
2. 相比于固定的空间衰减和注意力机制，动态调整可以更加精准地聚焦于图像中的关键信息，进一步提升模型的特征提
取和表达能力，有望在图像分类、目标检测等任务中取得更高的准确率。
"""

import torch
import torch.nn as nn
from typing import Tuple


class DynamicAdjustment(nn.Module):
    """动态空间自适应机制：计算衰减参数"""

    def __init__(self, embed_dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局池化
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 2)  # 输出两个动态调整参数
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        pooled = self.global_pool(x).view(b, c)  # 变为 (b, c)
        dynamic_params = self.fc(pooled)  # 计算动态参数
        return torch.sigmoid(dynamic_params)  # 归一化到 [0,1] 区间


class MaSA(nn.Module):
    def __init__(self, embed_dim, num_heads=4, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

        self.dynamic_adjust = DynamicAdjustment(embed_dim)  # 动态调整模块

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
        bsz, h, w, _ = x.size()

        dynamic_params = self.dynamic_adjust(x.permute(0, 3, 1, 2))  # 计算动态参数
        dynamic_factor_h, dynamic_factor_w = dynamic_params[:, 0], dynamic_params[:, 1]

        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)

        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        v = v.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        # 计算注意力权重（使用动态参数）
        attn_h = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * dynamic_factor_h[:, None, None, None, None],
                               dim=-1)
        attn_w = torch.softmax(
            torch.matmul(q.permute(0, 1, 3, 2, 4), k.permute(0, 1, 3, 2, 4).transpose(-1, -2)) * dynamic_factor_w[:,
                                                                                                 None, None, None,
                                                                                                 None], dim=-1)

        v = torch.matmul(attn_h, v)
        v = torch.matmul(attn_w, v.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)

        output = v.reshape(bsz, h, w, self.embed_dim * self.factor)  # 确保展平为正确的形状
        output = self.out_proj(output)

        return output.permute(0, 3, 1, 2)


if __name__ == "__main__":
    # 创建 MaSA 模块实例，输入通道数为 64
    module = MaSA(64)
    # 创建一个形状为 (1, 64, 128, 128) 的随机输入张量
    input_tensor = torch.randn(1, 64, 128, 128)
    # 通过 MaSA 模块计算输出
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())
    print('Output size:', output_tensor.size())
