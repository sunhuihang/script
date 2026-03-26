import torch
from torch import nn
'''
64. Multi-Scale and Detail-Enhanced Segment Anything Model for Salient 
Object Detection (ACM 2024 顶会论文)
即插即用模块：MSDE（多尺度细节增强模块）；MEEM（多尺度边缘增强模块）
一、背景
显著物体检测（SOD）旨在识别和分割图像中最突出的物体，在许多下游任务中发挥重要作用。过去十年，基于卷积神经网络（CNNs）
和 Transformer 的 SOD 方法取得了进展，但仍存在局限性。最近提出的 Segment Anything Model（SAM）具有强大的分割和
泛化能力，然而其在 SOD 应用中面临挑战，如需要准确的目标提示，且缺乏对多尺度、多层次信息的利用以及细粒度细节的融入。为
解决这些问题，本文提出了 Multi-scale and Detail-enhanced SAM（MDSAM），其中包含 MSDE 和 MEEM 两个关键模块。

二、MSDE 模块
1. 整体架构设计：MSDE 功能通过多个子模块协同实现，用于改进 SAM 以适应 SOD 任务。它主要由轻量级多尺度适配器（LMSA）、
多层次融合模块（MLFM）和细节增强模块（DEM）组成。LMSA 在几乎不增加训练参数的情况下使 SAM 能够学习多尺度信息；MLFM 
全面利用 SAM 编码器的多层次信息；DEM 则为 SOD 预测融入图像细节和边缘信息，共同提升模型性能。
2. 核心组件：
A. 轻量级多尺度适配器（LMSA）：在 SAM 编码器的每个 Transformer 层首次归一化前加入 LMSA。它先通过线性投影降低特征
维度，再利用平均池化获取多尺度特征，经深度卷积捕捉局部细节信息，最后融合多尺度特征并通过线性投影和残差连接输出，使SAM 
能高效适应 SOD 任务并有效利用多尺度信息。
B. 多层次融合模块（MLFM）：SAM 仅利用编码器最后一层输出，且简单的拼接融合策略无法充分整合多层次信息。MLFM 将 SAM 
编码器不同层的输出特征进行拼接，通过卷积得到聚合特征，基于此生成权重分配给不同层，进而融合得到充分整合多层次信息的特征，
为掩码解码器提供更丰富的输入。
C. 细节增强模块（DEM）：由于 SAM 的编码器图像补丁嵌入策略和解码器上采样策略会导致细节信息丢失，DEM 通过主、辅两个分
支来解决这一问题。主分支将掩码解码器的输出特征逐步上采样到输入分辨率，辅助分支利用多尺度边缘增强模块（MEEM）从输入图像
中提取细粒度细节信息，并添加到主分支，从而生成精确且细节丰富的 SOD 预测结果。
3. 微观设计考量：MSDE 功能通过三个模块的协同工作，实现了多尺度和细粒度信息的有效利用。LMSA 在保持训练高效和良好泛化能
力的同时，为模型引入多尺度信息；MLFM 确保了对 SAM 编码器多层次信息的充分利用；DEM 解决了 SAM 中细节信息缺失的问题，增
强了模型对复杂细节和边缘的捕捉能力，使 MDSAM 能够有效定位具有丰富细节信息的显著物体。

三、MEEM 模块
1. 整体架构设计：MEEM 作为 DEM 的辅助分支，用于从输入图像中提取多尺度边缘信息，并将其与 DEM 主分支特征融合，以增强模型对
细节的捕捉能力。
2. 核心组件：MEEM 首先使用 3×3 卷积层从输入图像中提取局部特征，然后通过一系列 1×1 卷积和 3×3 平均池化操作在不同尺度上提取
边缘信息。接着，利用边缘增强器（EE）突出特征图中的物体边缘。最后，将不同尺度的边缘增强特征通过通道拼接和 1×1 卷积进行融合，
得到包含多尺度边缘信息的特征，用于补充 DEM 主分支的特征。
3. 微观设计考量：MEEM 通过多尺度的操作提取边缘信息，能够在不同尺度上捕捉图像的细节特征。利用平均池化扩展感受野，降低计算复杂度，
同时边缘增强器的设计能够突出物体边缘，使模型更好地感知物体的轮廓和细节。通过将这些多尺度边缘信息与 DEM 主分支特征融合，有效
解决了 SAM 在处理复杂细节和边缘时的不足，提升了 MDSAM 在 SOD 任务中的性能。

四、适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
'''
class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class DetailEnhancement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(img_dim,feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act()
        )
        self.img_er = MEEM(feature_dim, feature_dim // 2, 4, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim *2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, padding=1, bias=False),
            norm(feature_dim * 2),
            act(),
        )

        self.out_conv = nn.Conv2d(feature_dim * 2, img_dim, 1)

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
        )

    def forward(self, img, feature, b_feature):
        feature = torch.cat([feature, b_feature], dim=1)
        feature = self.feature_upsample(feature)

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature

        out_feature = torch.cat([feature, img_feature], dim=1)
        out_feature = self.fusion_conv(out_feature)
        out = self.out_conv(out_feature)
        return out

# 输入 B C H W,  输出B C H W
if __name__ == "__main__":
    # 创建DetailEnhancement模块的实例
    # 第一个即插即用模块是多尺度细节增强模块：MSDE
    MSDE =DetailEnhancement(img_dim=64,feature_dim=128)
    input = torch.randn(1, 64, 128, 128)
    feature = torch.randn(1, 128, 32, 32)
    b_feature = torch.randn(1, 128, 32, 32)
    # 执行前向传播
    output= MSDE(input,feature,b_feature)
    print('MSDE Input size:', input.size())
    print('MSDE Output size:', output.size())
    print('---------------------------------')

    # 第二个即插即用模块是多尺度边缘增强模块：MEEM
    MEEM = MEEM(in_dim=64,hidden_dim=32)
    input = torch.randn(1, 64, 128, 128)
    # 执行前向传播
    output = MEEM(input)
    print('MEEM Input size:', input.size())
    print('MEEM Output size:', output.size())