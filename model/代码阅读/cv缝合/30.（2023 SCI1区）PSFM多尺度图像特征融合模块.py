import torch
import torch.nn as nn
'''
PSFM: 多尺度图像特征融合模块用于图像融合与语义理解 (2023, Information Fusion)
即插即用模块：PSFM多尺度融合模块（多功能模块）
一、背景
多模态图像融合（如红外与可见光图像）旨在集成源图像的互补特性，从而生成更具信息性和视觉吸引力的融合图像。
然而，大多数现有方法主要关注视觉质量，忽略了下游高层视觉任务的需求。针对这一不足，本文提出了一种基于渐
进语义注入与场景保真约束的图像融合网络PSFusion，其中包含一个多尺度特征融合模块（PSFM）。PSFM通过跨模
态特征的整合优化视觉质量，同时增强高层语义任务的适应性，有效提升了图像融合的性能。

二、PSFM模块原理
1. 输入特征：将不同模态的深层特征投影为查询（Q）、键（K）和值（V）向量，用于全局语义信息的融合。
2. 融合过程：
A. 特征增强：通过卷积和密集层对输入特征进行增强。
B. 注意力计算：以模态无关的查询向量Q为核心，分别与模态特定的K和V进行交互，生成语义丰富的融合特征。
3. 输出特征：融合后的多尺度特征既保留了不同模态的细节信息，也强化了全局上下文的理解。

三、适用任务
PSFM模块适用于以下任务和场景：
1. 图像融合：在红外与可见光等多模态图像融合任务中表现优异。
2. 语义分割与目标检测：能够提升下游语义任务的性能，特别是在复杂环境（如夜间、烟雾等）下表现突出。
3. 适用于其他语义分割和目标检测需要进行特征融合的任务。
'''
class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.Q = BBasicConv2d(in_C, out_C, 3, padding=1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, padding=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, padding=1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, 3, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class PSFM(nn.Module):
    def __init__(self, Channel):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)
        self.obj_fuse = GEFM(Channel * 2, Channel)

    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = PSFM(Channel=32)
    input1 = torch.rand(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)
    output = block([input1, input2])
    print('input1_size:', input1.size())
    print('input2_size:', input2.size())
    print('output_size:', output.size())
