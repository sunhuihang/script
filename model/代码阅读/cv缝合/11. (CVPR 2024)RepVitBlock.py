import torch.nn as nn
from timm.models.layers import SqueezeExcite
import torch
'''
RepViT: Revisiting Mobile CNN From ViT Perspective (CVPR 2024)
即插即用模块：RepViTBlock

一、背景：将轻量级卷积神经网络（CNN）的设计进行了改进，并引入了一系列模块，以提升其在移动设备上的表现。
轻量级ViT模型采用了“令牌混合器”（token mixer）和“通道混合器”（channel mixer）来分别处理空间和通道
信息，而传统CNN的卷积层则将两者耦合在一起。为了更高效地借鉴ViT的设计，论文对MobileNetV3的卷积块进行
改造。

二、RepViT模块
1.分离 Token Mixer 和 Channel Mixer：
RepViTBlock 模块将 token mixer（通过 3x3 深度卷积处理空间信息）和 channel mixer（通过 1x1 卷积处理通道信息）分离开来。
这种设计提高了空间信息与通道信息的处理效率，受到了 Vision Transformer (ViT) 架构的启发。
2.结构重参数化：
A. 分离令牌混合器和通道混合器：通过将3x3的深度卷积（Depthwise Convolution, DW）上移，以处理空间信息，并在它后面添加1x1
卷积来处理通道信息，从而实现空间信息和通道信息的解耦。
B. 结构重参数化（Structural Re-parameterization）：在训练过程中引入了多分支结构，提升模型的学习能力；但在推理时，重参数
化技术会将多分支结构合并成单一分支，减少计算开销。

三、适合任务：RepViTBlock 能在保持高准确率的同时，大幅度提升模型在移动设备上的推理效率，适用于图像分类、目标检测、实例分割、
语义分割等所有计算机视觉CV任务

'''

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2,hidden_dim=80, use_se=True, use_hs=True):
        super(RepViTBlock, self).__init__()
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64, 64)
    # 创建一个 RepViTBlock 模块实例
    model = RepViTBlock(inp=32,oup=32,kernel_size=3, stride=1)
    # 执行前向传播
    output = model(input)
    # 打印输入和输出的形状
    print('input_size:',input.size())
    print('output_size:',output.size())
