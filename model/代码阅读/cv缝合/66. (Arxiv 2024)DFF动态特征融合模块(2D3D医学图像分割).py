import torch
import torch.nn as nn

from timm.models.layers import DropPath
# B站：CV缝合救星
'''
66. D-Net: Dynamic Large Kernel with Dynamic Feature Fusion for Volumetric Medical Image Segmentation
即插即用模块：DFF(动态特征融合)

一、背景
在医学图像分割领域，Transformer 架构和卷积神经网络（CNNs）都有各自的应用。Transformer 中的分层 Transformer 因
大感受野和利用全局信息的能力取得了一定成果，但注意力机制存在计算复杂、难以有效提取局部信息的问题。CNNs 虽然擅长局部特
征提取，引入大卷积核可扩大感受野，但固定大小的内核使其难以捕获形状和大小差异大的器官的多尺度特征，也无法高效利用全局上
下文信息。为克服这些局限，本文提出动态大核（DLK）和动态特征融合（DFF）模块，并将它们集成到分层 Transformer 架构中，
构建了 D-Net*B站CV缝合救星*。

二、DFF 模块介绍
（一）整体设计
DFF 模块的设计目的是基于全局信息来自适应地融合多尺度局部特征。在融合过程中，它通过一系列操作，动态选择重要特征，以达到提
升模型对医学图像特征表达能力的效果*B站CV缝合救星*。。
（二）核心组件与操作
1. 特征拼接：把不同的特征映射按通道维度拼接起来，将多种特征信息整合，方便后续统一处理。
2. 通道调整与特征选择：为便于后续模块处理融合后的特征，需要减少通道数量。这一过程借助全局通道信息来引导。获取全局通道信
息时，先进行平均池化操作，再经过卷积层和 Sigmoid 激活，这样就能判断出特征的重要程度。根据这个信息对拼接后的特征进行校
准，然后用 1×1×1 卷积层筛选特征，留下重要的特征，去掉不重要的。
3. 利用空间信息校准特征：为了更好地把握*B站CV缝合救星*。局部特征图之间的空间关系，从之前的特征映射中，利用 1×1×1 卷积层和 Sigmoid 激活
获取全局空间信息。用这个信息校准经过通道调整后的特征图，突出重要的空间区域，从而得到最终融合好的特征。

三、微观设计考量
DFF 模块优势明显。它依据全局信息动态选择特征，比传统的融合方式更灵活。分别处理通道和空间信息，能更全面地挖掘特征里的关键
内容，提升特征质量。在处理不同尺度的医学图像特征时，该模块能平衡好局部和全局信息，避免信息丢失或过度关注某一方面，进而提
高模型整体性能。

四、适用任务
DFF 模块主要用于 3D 医学图像分割，在腹部多器官*B站CV缝合救星*。分割和多模态脑肿瘤分割任务中效果显著。在 D-Net 架构里，DFF 模块在编码器和
解码器之间负责融合特征。在腹部多器官分割时，帮助模型精准识别和分割不同器官；在多模态脑肿瘤分割中，能让模型更好地融合不同模
态的图像信息，提升对肿瘤区域的分割精度*B站CV缝合救星*。。实验显示，集成 DFF 模块的 D-Net 在这两个任务上的表现优于其他先进模型，证明了 DFF
 模块在医学图像分割任务中的有效性和实用性。
'''
class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output

if __name__ == '__main__':
    input1 = torch.randn(1, 32, 16, 64, 64) # x: (B, C, D,H, W) 3D图像维度
    input2 = torch.randn(1, 32, 16, 64, 64)  # x: (B, C, D,H, W) 3D图像维度
    model = DFF(32)
    output = model(input1,input2)
    print("DFF_input size:", input1.size())
    print("DFF_Output size:", output.size())