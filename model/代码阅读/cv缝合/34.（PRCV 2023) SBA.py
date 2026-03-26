import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')

'''
SBA模块：选择性边界聚合模块（Selective Boundary Aggregation Module）（PRCV 2023）
即插即用模块：SBA（强化模块）
一、背景
在医学图像分割任务中，基于 Transformer 的方法虽然在捕捉长距离依赖方面表现出色，但容易丢失局部特征，
导致对小物体的预测过于平滑以及物体边界模糊。同时，传统的融合低层次边界信息和高层次语义信息的方法存在
冗余和不一致的问题。为解决这些问题，本文提出了SBA模块，旨在通过选择性地聚合边界信息和语义信息，
更好地描绘物体的精细轮廓并定位重新校准的物体。
二、SBA 模块原理
1. 输入特征：SBA 模块接收来自编码器不同层次的浅层和深层特征，浅层特征富含边界细节但语义较少，深层特征
则包含丰富的语义信息。
2. 重新校准注意力单元（RAU）：
A. 设计了新颖的 RAU 块，在融合之前自适应地从两个输入（浅层特征和深层特征）中提取相互表示。
B. 通过两个线性映射和 sigmoid 函数将输入特征的通道维度减少到 32，得到特征图和，并进行逐点相乘等操作。
3. 特征融合：
A. 浅层和深层信息通过不同方式输入到两个RAU块，以弥补高层语义特征缺失的空间边界信息和低层特征缺失的语
义信息。
B. 两个RAU块的输出经过 3×3 卷积后进行拼接，实现不同特征的稳健组合并细化粗糙特征。
4. 输出特征：输出包含增强边界信息和语义表示的特征图，用于后续的图像分割任务。
三、适用任务
1. 医学图像分割：尤其适用于解决医学图像中物体边界不清晰、小物体分割困难等问题，如结肠镜图像中的息肉分割和皮肤镜图
像中的皮肤病变分割等。
2. 其他对边界精度和小物体分割要求较高的图像分割任务：通过选择性地聚合边界和语义信息，能够提升分割的准确性和精细度。
'''
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Block(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        if bn_start:
            self.add_module('norm1', norm_layer(input_num)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', norm_layer(num1)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                           dilation=dilation_rate, padding=dilation_rate)),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(Block, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature
def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class SBA(nn.Module):

    def __init__(self, input_dim=64,output_dim = 64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = BasicConv2d(input_dim // 2, input_dim // 2, 1)

        self.conv = nn.Sequential(BasicConv2d(input_dim, input_dim, 3, 1, 1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))
        self.fc1 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, H_feature, L_feature):
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature,
                                                                                       size=L_feature.size()[2:],
                                                                                       align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature,
                                                                                       size=H_feature.size()[2:],
                                                                                       align_corners=False)

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64) # x: (B, C,H, W)
    input2 = torch.randn(1, 32, 64, 64) # x: (B, C,H, W)
    model = SBA(input_dim=32,output_dim=32)
    output = model(input1,input2)
    print("SBA_input size:", input1.size())
    print("SBA_Output size:", output.size())