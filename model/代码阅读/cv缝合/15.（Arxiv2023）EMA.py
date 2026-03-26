import torch
from torch import nn
"""
高效多尺度注意力模块 EMA (Efficient Multi-Scale Attention) (Arxiv 2023)

一、背景：
随着深度学习的发展，CNN 已被广泛应用于图像分类和目标检测任务。通过堆叠多个卷积层，CNN可以捕获不同尺度和复杂程度的特征，
显著提升了模型的特征表示能力。然而，随着网络深度的增加，出现了两个主要挑战：
1. 高计算成本和内存需求：每增加一个卷积层，模型参数和计算量都会相应增加。这不仅消耗大量的内存资源，也显著增加了计算成本。
2. 模型复杂性增加：更深的网络结构虽然增强了特征提取能力，但也引入了大量的冗余特征和噪声，降低了模型的学习效率。
为了解决这个问题，注意力机制逐渐被引入到深度学习中，通过选择性地关注特定的特征维度来减少计算需求。EMA聚焦于通道和空间维
度的跨维度注意力交互，从而在降低计算开销的同时保留特征信息。

二、EMA原理机制
1. 输入特征分组：EMA 模块首先将输入特征图的通道维度分成 g 个子组，每个子组的通道数为c//g。
2. 多尺度卷积分支:
A. 多分支结构：征EMA 使用两个并行分支来处理每个子组的特。一个分支应用1×1 卷积核（以保留原始特征），另一个分支应用3×3
卷积核（用于捕获多尺度的局部信息）。通过多尺度卷积，EMA可以从不同的尺度提取特征，从而增强特征图的表示能力。
B. 特征聚合：每个卷积分支生成的特征图经过处理后会被组合（Concat + 1x1 Conv），以聚合不同卷积核提取的特征，
从而同时保留局部和全局信息。
3. 跨空间注意力权重生成：
A. 空间平均池化：在每个并行分支中，分别对特征图进行全局平均池化（Avg Pooling），计算特征在水平（X Avg Pool）和垂直
（Y Avg Pool）方向的平均值，以获得不同空间维度的信息。
B. 权重计算：对于每个方向的池化结果，分别通过 Sigmoid 和 Softmax 激活函数，得到注意力权重图。Sigmoid 用于对每个子组
的权重进行归一化，而 Softmax 则用于生成跨通道的注意力权重。
C. 矩阵相乘 (Matmul)：权重图与特征图进行矩阵相乘，以生成最终的加权特征图，这样可以强化不同空间位置的显著特征。
4. 输出特征融合:
两个分支的注意力权重图经过加权相乘后，通过一个 1×h×w 的特征图汇总（加法操作），接着再通过一个 Sigmoid激活
函数进行归一化，得到最终的注意力特征图。

三、适用任务：适用于图像分类、分割、小目标检测、图像恢复等所有CV任务。
"""

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# 输入 N C HW,  输出 N C H W
if __name__ == '__main__':
    block = EMA(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print('input_size:',input.size())
    print('output_size:',output.size())
