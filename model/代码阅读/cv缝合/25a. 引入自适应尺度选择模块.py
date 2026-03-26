import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
"""
CV缝合救星魔改创新1：自适应尺度选择模块
1. 动态尺度选择：根据输入图像的尺寸选择合适的卷积尺度。如果图像较大，使用较大的尺度（例如7x7），以捕捉更多的全局信息；
如果图像较小，则使用较小的尺度（例如3x3），以提高计算效率。
2. 简化计算：通过动态调整使用的尺度，避免了在不需要大尺度的任务中进行大量计算。
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


# 自适应尺度选择模块
class AdaptiveScaleMLA(nn.Module):
    """自适应尺度选择模块，动态选择尺度以适应不同的输入图像"""

    def __init__(self, in_channels: int, out_channels: int, base_scales: Tuple[int, ...] = (3, 5, 7),
                 use_bias=False, norm="bn2d", act_func="relu", eps=1e-5):
        super(AdaptiveScaleMLA, self).__init__()
        self.eps = eps
        self.base_scales = base_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 基于不同尺度构建卷积层
        self.scales_conv = nn.ModuleList([
            ConvLayer(in_channels, out_channels, kernel_size=scale, use_bias=use_bias, norm=norm, act_func=act_func)
            for scale in self.base_scales
        ])

    def select_scale(self, x: torch.Tensor) -> torch.Tensor:
        """动态选择合适的尺度"""
        # 计算图像的尺寸，动态选择尺度
        H, W = x.shape[2], x.shape[3]
        if H * W > 1024 * 1024:
            selected_scale = self.base_scales[-1]  # 大尺寸选择最大的尺度
        elif H * W > 512 * 512:
            selected_scale = self.base_scales[1]  # 中等尺寸选择中间的尺度
        else:
            selected_scale = self.base_scales[0]  # 小尺寸选择最小的尺度
        return selected_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 根据输入图像的尺寸选择合适的尺度
        selected_scale = self.select_scale(x)
        # 获取对应尺度的卷积层
        scale_idx = self.base_scales.index(selected_scale)
        conv_layer = self.scales_conv[scale_idx]
        return conv_layer(x)


# 测试自适应尺度选择模块
if __name__ == '__main__':
    block = AdaptiveScaleMLA(in_channels=64, out_channels=64)  # 通过选择不同的卷积尺度
    input1 = torch.rand(3, 64, 32, 32)  # 输入尺寸为 32x32
    output = block(input1)
    print(input1.size())
    print(output.size())
    input2 = torch.rand(3, 64, 256, 256)  # 输入尺寸为 256x256
    output = block(input2)
    print(input2.size())
    print(output.size())
