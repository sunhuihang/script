import torch
import torch.nn as nn
import torch.nn.functional as F
"""
B站CV缝合救星魔改创新：DynamicScope SSA+
1. DynamicSSA 类：用于实现动态核大小调整。通过全局平均池化、全连接层和 Sigmoid
 函数计算出一个动态的核大小，根据输入图像的特征自适应调整注意力计算的范围。
2. SSA 类改进：增加了多尺度注意力融合功能，接受多个核大小的输入，对不同核大小的注
意力结果进行平均融合，以增强模型对不同尺度特征的捕捉能力。
3. spatial_strip_att_unit 类：保持原有的空间带状注意力单元的功能，用于计算每个
位置的注意力权重和特征聚合。
"""

class SSA(nn.Module):
    def __init__(self, dim, group, kernel_sizes) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))
        self.ssa_modules = nn.ModuleList([
            spatial_strip_att_unit(dim, kernel=kernel, group=group, H=True) for kernel in kernel_sizes
        ] + [
            spatial_strip_att_unit(dim, kernel=kernel, group=group, H=False) for kernel in kernel_sizes
        ])

    def forward(self, x):
        multi_scale_outputs = []
        for module in self.ssa_modules:
            out = module(x)
            multi_scale_outputs.append(out)
        out = torch.mean(torch.stack(multi_scale_outputs), dim=0)
        return self.gamma * out + x * self.beta


class spatial_strip_att_unit(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)

        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class DynamicSSA(nn.Module):
    def __init__(self, dim, group, min_kernel=3, max_kernel=7):
        super().__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.dim = dim
        self.group = group
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 16, 1, bias=False),
            nn.Sigmoid()
        )
        self.ssa = SSA(dim, group, [min_kernel, max_kernel])

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1, 1)
        kernel_size = int(self.min_kernel + y.item() * (self.max_kernel - self.min_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        new_ssa = SSA(self.dim, self.group, [kernel_size])
        return new_ssa(x)


if __name__ == "__main__":
    # 模块参数
    batch_size = 1
    channels = 32
    height = 256
    width = 256

    # 创建DynamicSSA模块
    dynamic_ssa = DynamicSSA(dim=32, group=1)
    print(dynamic_ssa)
    print("B站CV缝合救星创新的SSA模块, 增加动态核大小调整和多尺度注意力融合功能!")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 打印输入张量的形状
    print("Input shape:", x.shape)

    # 前向传播计算输出
    output = dynamic_ssa(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)