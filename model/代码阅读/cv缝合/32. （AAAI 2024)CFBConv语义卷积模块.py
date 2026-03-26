from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_ as trunc_normal_init
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


'''
CFBConv: 卷积滤波基元用于图像处理与特征提取（2024, CVPR）
即插即用模块：CFBConv
一、背景
在深度学习领域，卷积神经网络（CNN）在图像处理和视觉任务中取得了显著成效，但仍面临一些局限，
如卷积核固定、对不同尺度信息处理不够灵活等问题。为了解决这些挑战，本文提出了一种新的卷积操
作——CFBConv（Convolutional Filter Bank Convolution）。该方法通过引入卷积滤波基元，
增强了卷积神经网络在不同尺度和不同特征层次上的表现，尤其在图像多尺度建模和特征融合方面具有
明显优势。

二、CFBConv模块原理
1. 输入特征：CFBConv模块接收输入图像或特征图，并通过自适应卷积滤波器进行处理。
2. 融合过程：
A. 滤波基元生成：CFBConv使用多个滤波器组合构建卷积基元，其中每个基元负责捕捉图像中的不同频率特征。
通过学习不同尺度和方向的卷积核，模型能够更有效地提取局部和全局信息。
B. 多尺度卷积：基于滤波基元的设计，CFBConv能够同时处理多尺度信息，自动适应图像的不同尺寸和复杂结构。 
C. 自适应滤波器更新：CFBConv模块的滤波器根据输入数据自适应更新，动态调整以适应不同的图像特征和任务需求。
3. 输出特征：CFBConv通过多尺度卷积操作，输出的特征图包含了丰富的空间信息，并且能够捕捉不同尺度的特征，
提升了图像理解的精度和鲁棒性。

三、适用任务
1. 图像分类：特别是在复杂背景或多尺度物体分类任务中，CFBConv能显著提升性能。
2. 目标检测与实例分割：在小物体检测和高分辨率目标识别任务中具有良好的表现。
3. 多尺度图像理解：能够高效处理各种尺度的输入图像，适用于卫星图像、医学影像等复杂数据的分析。
4. 视频分析与动态物体追踪：支持从时序数据中提取有意义的动态特征，适用于视频分析和动态物体追踪等任务。
5. 图像生成与超分辨率重建：在图像生成与超分辨率任务中，通过多尺度卷积提升生成图像的细节和清晰度。
'''

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP,self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)  #TODO,1e-6?
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
class ConvolutionalAttention(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001)
        trunc_normal_init(self.kv3, std=0.001)


    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, h, w])
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w)
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        x1 = F.conv2d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=(3,0))
        x1 = self._act_dn(x1)
        x1 = F.conv2d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=(3,0))
        x3 = F.conv2d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=(0,3))
        x3 = self._act_dn(x3)
        x3 = F.conv2d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,3))
        x=x1+x3
        return x
class CFBConv(nn.Module):
    """
    The CFBConv implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.):
        super(CFBConv,self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = ConvolutionalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x)) 
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    #：
    models = CFBConv(in_channels=32,out_channels=32).cuda()
    input = torch.randn(1, 32, 64, 64).cuda()
    output = models(input)
    print('input_size:',input.size())
    print('output_size:',output.size())