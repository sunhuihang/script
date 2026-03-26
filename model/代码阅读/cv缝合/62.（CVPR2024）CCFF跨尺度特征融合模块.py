import torch
import torch.nn as nn
import torch.nn.functional as F
'''
62. DETRsBeat YOLOsonReal-time Object Detection （CVPR 2024）
即插即用模块：CCFF（跨尺度特征融合模块）
一、背景
在实时目标检测领域，基于 CNN的YOLO系列因速度和精度的合理权衡而广受欢迎，但非极大值抑制（NMS）
影响其速度和精度，且不同场景需谨慎选择 NMS 阈值。基于 Transformer 的端到端检测器（DETRs）
虽有优势，但计算成本高，限制了其实时应用。本文旨在提出创新模块提升实时目标检测性能。

二、CCFF 原理
1. 整体架构设计：CCFF 是 RT-DETR 中高效混合编码器的组成部分，与基于注意力的层内特征交互（AIFI）
模块协同工作。其输入为骨干网络最后三个阶段的特征，AIFI 对进行层内特征交互，CCFF 则负责将、与 AIFI
 处理后的相关特征进行跨尺度特征融合，将多尺度特征转换为图像特征序列，为后续的解码器提供支持。
2. 卷积调制块核心组件：CCFF 包含多个融合块，每个融合块由卷积层构成。融合块通过两个 1×1 卷积调整通
道数，利用 N 个由 RepConv 构成的 RepBlock 进行特征融合，最后通过逐元素相加融合两路输出，以此实现
相邻尺度特征的融合。
3. 微观设计考量：CCFF 通过这种设计优化跨尺度特征融合，在融合过程中有效整合不同尺度特征的信息，增强特
征的表达能力，为模型在目标检测任务中提供更丰富、更具代表性的特征，有助于提高检测精度。

三、适用任务
1. 目标检测：在 RT-DETR 中应用 CCFF 模块，RT-DETR-R50 在 COCO 数据集上达到 53.1% 的平均精度均值
（AP），RT-DETR-R101 达到 54.3% AP，相比之前先进的 YOLO 检测器和相同骨干网络的 DETRs，在速度和精
度上均有显著提升，表明 CCFF 模块对目标检测性能的积极影响。
2. 模型缩放：RT-DETR 支持灵活缩放，CCFF 模块在其中发挥作用。通过调整相关组件参数，RT-DETR 能设计出
不同规模的模型，在不同场景下均有出色表现，如缩放后的 RT-DETR 在与较轻量级的 YOLO 检测器比较时，速度
和精度上都更胜一筹，体现 CCFF 模块在不同规模模型中的适用性和有效性。
3. 目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
'''

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


class CCFF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CCFF, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x1, x2):
        x_1 = self.conv1(x1)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x2)
        return self.conv3(x_1 + x_2)

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 实例化模型对象
    model = CCFF(in_channels=64, out_channels=64)
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)
    output = model(input1,input2)
    print('input1_size:',input1.size())
    print('input2_size:', input2.size())
    print('output_size:',output.size())
