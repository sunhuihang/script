import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite
"""
CV缝合救星魔改2：多尺度SERepVit
缺点：SE层无法适应多尺度特征
当前的 RepViTBlock 通过交替放置 SE（Squeeze-and-Excite）层来处理通道注意力，虽然节省了计算资源，但这种交替放置的方法在捕获多尺度特征方
面有一定限制。在复杂场景中，不同尺度的特征往往包含不同的关键信息，仅在局部区域应用SE层可能导致对大尺度特征的注意力不足，影响模型对复杂场景的
适应性。

改进方法
为了增强模型对多尺度特征的适应性，设计了一个多尺度SE模块 MultiScaleSE。通过在全局和局部尺度上分别生成SE权重，然后将这些权重整合，增强SE层
对多尺度特征的捕获能力。这样，模型可以更好地对不同特征尺度进行权重调整，从而提升模型在处理复杂场景中的性能。
"""

# 带Batch Normalization的卷积层
class Conv2d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn_weight_init=1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


# 多尺度SE模块
class MultiScaleSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(MultiScaleSE, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d(2)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局和局部池化
        global_se = self.global_pool(x)
        local_se = self.local_pool(x)
        # 上采样局部池化结果并叠加
        se = global_se + nn.functional.interpolate(local_se, size=global_se.shape[-2:])
        se = self.fc1(se)
        se = nn.ReLU()(se)
        se = self.sigmoid(self.fc2(se))
        return x * se


# 带多尺度SE层的RepViT块
class MultiScaleSERepViTBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2, hidden_dim=80, use_se=True, use_hs=True):
        super(MultiScaleSERepViTBlock, self).__init__()

        # Token Mixer：3x3 DW卷积 + 多尺度SE层
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=inp),
                MultiScaleSE(inp) if use_se else nn.Identity(),  # 多尺度SE层
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=inp),
                MultiScaleSE(inp) if use_se else nn.Identity(),  # 多尺度SE层
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )

        # Channel Mixer
        self.channel_mixer = nn.Sequential(
            Conv2d_BN(oup, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU() if use_hs else nn.ReLU(),
            Conv2d_BN(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bn_weight_init=0),
        )

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x


# 测试代码
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input_tensor = torch.randn(1, 32, 64, 64)
    # 创建一个 MultiScaleSERepViTBlock 模块实例
    model = MultiScaleSERepViTBlock(inp=32, oup=32, kernel_size=3, stride=2)
    # 执行前向传播
    output_tensor = model(input_tensor)
    # 打印输入和输出的形状
    print('MultiScaleSERepViTBlock - input size:', input_tensor.size())
    print('MultiScaleSERepViTBlock - output size:', output_tensor.size())
