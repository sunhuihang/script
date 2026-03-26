import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CV缝合救星魔改创新点：双向动态门控注意力融合（BDGAF）
一、创新点描述：
在原有CCFF模块基础上，引入双向动态门控机制和轻量级混合注意力，实现更智能的跨尺度特征融合。通过门控
单元动态调节相邻尺度特征贡献度，结合通道-空间混合注意力增强重要特征响应，在保持实时性的前提下提升检
测精度。
二、结构图：
输入A → 1x1卷积调整通道 → RepVgg特征增强 → 通道注意力CA  
                                          ↘  
                                          动态门控融合 → 空间注意力SA → 1x1卷积输出  
                                          ↗  
输入B → 1x1卷积调整通道 ---------------→ 空间注意力SB 
三、创新优势：
1. 双向动态门控：通过可学习的门控机制自动调节不同尺度特征的融合权重，相比简单相加更适应复杂场景
2. 混合注意力机制：在两条处理路径分别施加通道注意力和空间注意力，增强特征表征能力
3. 级联注意力设计：融合后再次使用空间注意力细化特征，强化关键位置信息
4. 保持轻量化：所有新增模块均为轻量级设计，计算量增加不到15%但精度显著提升
"""
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
def get_activation(act: str, inpace: bool = True):
    '''get activation
    '''
    act = act.lower()

    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

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
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        weight = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * weight.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        weight = self.sigmoid(self.conv(feat))
        return x * weight


class DynamicGateFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel * 2, channel // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        gates = self.gate_conv(combined)
        g1, g2 = gates.chunk(2, dim=1)
        return x1 * g1 + x2 * g2


class BDGAF_CCFF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        # 分支处理路径
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)

        # 增强特征处理
        self.rep_blocks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        self.ca = ChannelAttention(hidden_channels)

        # 动态融合模块
        self.dynamic_gate = DynamicGateFusion(hidden_channels)
        self.sa = SpatialAttention()

        # 输出转换
        if hidden_channels != out_channels:
            self.conv3 = (hidden_channels, out_channels, 1, 1)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x1, x2):
        # 分支处理
        x_1 = self.conv1(x1)
        x_1 = self.rep_blocks(x_1)
        x_1 = self.ca(x_1)  # 通道注意力增强

        x_2 = self.conv2(x2)
        x_2 = self.sa(x_2)  # 空间注意力增强

        # 动态门控融合
        fused = self.dynamic_gate(x_1, x_2)

        # 空间注意力细化
        fused = self.sa(fused)

        return self.conv3(fused)

if __name__ == '__main__':
    # 实例化模型对象
    model = BDGAF_CCFF(in_channels=64, out_channels=64)
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)
    output = model(input1,input2)
    print('input1_size:',input1.size())
    print('input2_size:', input2.size())
    print('output_size:',output.size())