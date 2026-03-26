import torch
import torch.nn as nn
"""
CV缝合救星魔改2:引入轻量化卷积结构

一、缺点：在一些计算资源受限的场景（如移动设备）中，使用标准卷积层可能会增加计算负担。
二、CV缝合救星魔改：引入轻量化卷积结构（Depthwise Separable Convolutions）来降低计算成本，同时保持模型的表达能力。
轻量化卷积结构将传统卷积操作分解为深度卷积和逐点卷积，大幅减少参数量和计算量。
"""

class InceptionDWConv2dLightweight(nn.Module):
    """ 轻量化卷积的大核分解卷积模块 """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=9, branch_ratio=0.125):
        """
        Args:
            in_channels (int): 输入通道数
            square_kernel_size (int): 方形卷积核大小（默认值为3）
            band_kernel_size (int): 带状卷积核大小（默认值为9）
            branch_ratio (float): 分配给每个卷积分支的通道比例（默认值为0.125）
        """
        super().__init__()

        gc = int(in_channels * branch_ratio)  # 每个卷积分支的通道数

        # 使用轻量化的 Depthwise Separable 卷积
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),  # Depthwise
            nn.Conv2d(gc, gc, kernel_size=1)  # Pointwise
        )
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            # Depthwise
            nn.Conv2d(gc, gc, kernel_size=1)  # Pointwise
        )
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            # Depthwise
            nn.Conv2d(gc, gc, kernel_size=1)  # Pointwise
        )

        # 记录各分支的通道数，以便在前向传播时切分输入张量
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        # 将输入张量按通道数切分为身份映射分支、方形卷积分支、带状卷积分支（宽）和带状卷积分支（高）
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        # 将各分支的输出拼接在一起
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


# 测试模块
if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 64, 64)  # 创建一个示例输入张量
    model = InceptionDWConv2dLightweight(in_channels=32)  # 创建轻量化卷积的 IDC 模块
    output = model(input_tensor)
    print('输入大小:', input_tensor.size())
    print('输出大小:', output.size())
