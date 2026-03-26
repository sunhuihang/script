import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(act: str, inplace: bool = True):
    act = act.lower() if isinstance(act, str) else None
    act_map = {
        'silu': nn.SiLU,
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'gelu': nn.GELU
    }
    if act is None:
        return nn.Identity()
    elif isinstance(act, nn.Module):
        return act
    elif act in act_map:
        return act_map[act](inplace=inplace)
    else:
        raise ValueError(f"Unknown activation: {act}")

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x))
        return x * scale

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return F.pad(kernel1x1, [1, 1, 1, 1]) if kernel1x1 is not None else 0

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - running_mean * gamma / std

class GatedCCFF(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu", use_se=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)

        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])

        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
            nn.Sigmoid()
        )

        self.use_se = use_se
        self.se = SEBlock(hidden_channels) if use_se else nn.Identity()

        self.out_proj = nn.Identity() if hidden_channels == out_channels else \
                        ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        x1 = self.bottlenecks(x1)
        x_cat = torch.cat([x1, x2], dim=1)
        gate = self.gate(x_cat)
        fused = x1 * gate + x2 * (1 - gate)

        fused = self.se(fused)
        return self.out_proj(fused)

# 测试模块可用性
if __name__ == '__main__':
    model = GatedCCFF(in_channels=64, out_channels=64)
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    y = model(x1, x2)
    print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}, output shape: {y.shape}")
