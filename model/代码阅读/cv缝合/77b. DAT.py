import torch
import torch.nn as nn


class DynamicAdaptiveTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last

        # 为每个通道单独学习 alpha 参数
        self.alpha = nn.Parameter(torch.ones(normalized_shape))
        # 为每个通道单独学习偏置参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 用于动态调整 alpha 的缩放因子
        self.alpha_scale = nn.Parameter(torch.ones(1))
        # 用于动态调整偏置的缩放因子
        self.bias_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 计算输入数据的均值和标准差
        if self.channels_last:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            std = torch.std(x, dim=(0, 2, 3), keepdim=True)
        else:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            std = torch.std(x, dim=(0, 2, 3), keepdim=True)

        # 根据输入数据的统计特征动态调整 alpha 和偏置
        dynamic_alpha = self.alpha_scale * self.alpha.view(1, -1, 1, 1) / (std + 1e-8)
        dynamic_bias = self.bias_scale * self.bias.view(1, -1, 1, 1) * mean

        x = torch.tanh(dynamic_alpha * x + dynamic_bias)

        if self.channels_last:
            return x
        else:
            return x


if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 128, 128)
    # 创建 DynamicAdaptiveTanh 模块实例
    module = DynamicAdaptiveTanh([32])
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())
    print('Output size:', output_tensor.size())