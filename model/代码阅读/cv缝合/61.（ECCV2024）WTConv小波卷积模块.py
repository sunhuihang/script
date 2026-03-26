import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data

'''
61. Wavelet Convolutions for Large Receptive Fields ECCV 2024
即插即用模块：WTConv（替身模块）

一、背景
卷积神经网络（CNNs）在计算机视觉领域曾占据主导地位，但随着视觉 Transformer（ViTs）的出现，其面临竞争。
ViTs 的多头自注意力层能实现全局特征混合，而 CNNs 的卷积操作局限于局部特征混合。为缩小两者性能差距，一些
研究尝试增加卷积核大小，但该方法存在参数过多、性能饱和等问题。本文提出 WTConv 模块，旨在利用小波变换（WT）
有效扩大卷积的感受野，同时避免过参数化。

二、WTConv 原理
1. 整体架构设计：WTConv 利用小波变换的级联分解，对输入的不同频率带进行一系列小核卷积。先使用 WT 对输入的
低频和高频内容进行滤波和下采样，然后在不同频率图上进行小核深度卷积，最后通过逆小波变换（IWT）构建输出。通过
这种方式，使小核卷积能在更大的输入区域上操作，扩大感受野。
2. 卷积调制块核心组件
A. 小波变换与卷积结合：采用 Haar 小波变换，其一级变换通过与特定卷积核进行深度卷积并下采样实现。在不同频率
带上进行小核卷积，分离卷积操作，让小核在更大输入区域发挥作用。
B. 多频率融合：通过线性操作的性质，将不同频率层卷积的结果进行融合。具体是利用小波变换及其逆变换的线性特性，
将各级卷积结果相加，得到最终输出。
3. 微观设计考量
A. 扩大感受野与控制参数：随着小波变换层级增加，感受野呈指数增长，而训练参数仅线性增加，有效扩大感受野的同时
避免过参数化。
B. 增强低频响应：重复的小波分解使模块更注重输入的低频部分，相比标准卷积能更好地捕捉低频信息，提高对形状的响
应能力。
C. 计算成本优势：在实现类似感受野时，相比传统大核卷积，WTConv 的计算成本（FLOPs）更低。

三、适用任务
1. 图像分类：在 ConvNeXt 架构中使用 WTConv 替代 7×7 深度卷积，在 ImageNet-1K 分类任务中，相比其他增加
感受野的方法，在参数效率和分类准确率上表现更优。
2. 语义分割：以 WTConvNeXt 为骨干网络用于 UperNet 进行 ADE20K 语义分割任务，相比原 ConvNeXt 骨干网络，
平均交并比（mIoU）有所提升。
3. 目标检测：将 WTConvNeXt 作为 Cascade Mask R-CNN 的骨干网络在 COCO 数据集上进行目标检测，边界框和掩
码的平均精度（APbox 和 APmask）均有显著提高。
4. 其他优势体现任务：在模型可扩展性、对图像损坏的鲁棒性以及对形状的响应优于纹理等方面，WTConv 都展现出优势，
在相关评估实验中得到验证。
'''

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = WTConv2d(32,32)
    input = torch.rand(1, 32, 64, 64)
    output = block(input)
    print("input.shape:", input.shape)
    print("output.shape:",output.shape)