import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
"""
CV缝合救星魔改创新2：加权多尺度融合
1. 在原始的多尺度处理方法中，通常每个尺度的贡献是相等的。通过加权多尺度融合，可以为不同尺度的特征分配不同的
权重，确保更重要的尺度对输出的贡献更大。我们通过一个可学习的参数（即权重）来为每个尺度的卷积输出分配一个权重。
2. 实现思路：
A. 为每个尺度学习权重：我们为每个尺度的输出引入一个学习参数 scale_weights，它的大小与尺度数目相同。
B. 加权融合：每个尺度的输出将乘以相应的权重，最后加权求和得到最终的输出。
"""

# 获取与卷积核大小相同的填充
def get_same_padding(kernel_size: int or Tuple[int, ...]) -> int or Tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


# 激活函数构建
def build_act(name: str, **kwargs) -> nn.Module or None:
    act_dict = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    if name in act_dict:
        return act_dict[name](**kwargs)
    return None


# 卷积层
class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, dilation=1, groups=1,
                 use_bias=False, dropout=0, norm="bn2d", act_func="relu"):
        super(ConvLayer, self).__init__()
        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride), padding=padding,
                              dilation=(dilation, dilation), groups=groups, bias=use_bias)
        self.norm = build_act(norm, num_features=out_channels) if norm else None
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# 加权多尺度融合模块
class WeightedMultiScaleMLA(nn.Module):
    """加权多尺度融合模块"""

    def __init__(self, in_channels: int, out_channels: int, base_scales: Tuple[int, ...] = (3, 5, 7),
                 use_bias=False, norm="bn2d", act_func="relu"):
        super(WeightedMultiScaleMLA, self).__init__()

        self.base_scales = base_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 基于不同尺度构建卷积层
        self.scales_conv = nn.ModuleList([
            ConvLayer(in_channels, out_channels, kernel_size=scale, use_bias=use_bias, norm=norm, act_func=act_func)
            for scale in self.base_scales
        ])

        # 为每个尺度学习一个权重
        self.scale_weights = nn.Parameter(torch.ones(len(self.base_scales)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算每个尺度的输出
        scale_outputs = [conv_layer(x) for conv_layer in self.scales_conv]

        # 对每个尺度的输出应用权重
        weighted_output = sum(w * scale_output for w, scale_output in zip(self.scale_weights, scale_outputs))
        return weighted_output


# 测试加权多尺度融合模块
if __name__ == '__main__':
    block = WeightedMultiScaleMLA(in_channels=64, out_channels=64)
    input1 = torch.rand(3, 64, 32, 32)  # 输入尺寸为 32x32
    output = block(input1)
    print(input1.size())
    print(output.size())

    input2 = torch.rand(3, 64, 256, 256)  # 输入尺寸为 256x256
    output = block(input2)
    print(input2.size())
    print(output.size())
