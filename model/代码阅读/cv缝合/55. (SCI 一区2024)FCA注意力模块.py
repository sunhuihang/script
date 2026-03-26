import math
import torch
from torch import nn
'''
55. 用于图像去雾的无监督双向对比重建和自适应细粒度信道注意力网络 (SCI 一区 2024 顶刊 Neural Networks) 
自适应细粒度通道注意力（FCA）模块：实现高效特征权重分配提升图像去雾性能
一、背景
1. 图像去雾注意力机制困境：在图像去雾领域，SE 通道注意力虽被广泛应用，但仅用全连接层捕获全局信息，
忽视局部信息，致使特征权重分配不合理，影响去雾效果。当前缺乏能有效整合全局和局部信息以合理分配权重
的通道注意力机制。
2. FCA设计动机：为克服此问题，FCA 应运而生，旨在利用相关矩阵捕捉不同粒度下的全局和局部信息关联，
促进二者交互，实现更高效的特征权重分配，提升图像去雾网络性能。
二、模块原理
1. 特征转换与局部信息获取：对于输入的特征图，先经全局平均池化将其转换为通道描述符。再利用带状矩阵
进行局部通道交互获取局部信息，同时用对角矩阵捕获全局信息。
2. 信息关联与权重分配：通过交叉相关操作获取关联矩阵，捕捉不同粒度下全局和局部信息的相关性。接着提
取信息作为权重向量，经可学习因子动态融合。最后将所得权重与输入特征图相乘得到最终输出特征图。

三、适用场景：图像恢复，图像去噪、雨、雪、雾，目标检测，图像增强等所有CV2二维任务通用。
'''
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class FCAttention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input*out

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1,64,256,256)
    model = FCAttention(channel=64)
    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())
