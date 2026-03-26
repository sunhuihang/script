import torch
import torch.nn as nn
'''
CGLU: 卷积门控通道注意力机制用于视觉感知与特征融合 (2024, CVPR)
即插即用模块：CGLU（替身模块）

一、背景
Vision Transformers (ViTs)近年来在各种视觉任务中表现突出，但仍面临信息混合不足的问题，尤其是在深层网络中容易出现深度
退化效应。针对这一问题，本文提出了一种结合卷积和门控线性单元（GLU）的通道混合器——CGLU（Convolutional Gated Linear 
Unit），旨在增强局部建模能力并提高模型的鲁棒性。通过与新型注意力机制（如像素聚焦注意力）结合，CGLU显著改善了视觉模型的
性能。

二、CGLU模块原理
1. 输入特征：提取输入图像的通道特征，通过卷积和门控机制进行处理。
2. 融合过程：
A. 卷积操作：引入轻量级的3×3深度卷积，增强局部信息捕获，同时提供条件位置编码。
B. 门控机制：使用门控线性单元（GLU）控制信息流动，每个token的门控信号由其最近邻的特征生成。
3. 输出特征：生成的特征既包含丰富的全局上下文信息，也保留了局部细节特征，提升了模型对噪声和扰动的鲁棒性。

三、适用任务
CGLU模块适用于以下任务和场景：
1. 图像分类：在ImageNet等大型图像分类任务中表现优异。
2. 目标检测与语义分割：特别是在复杂场景下（如夜间、烟雾环境）显著提高了性能。
3. 多尺度推理：支持高效的多尺度图像推理，性能优于传统卷积模型。
4. 其它视觉任务：适合需要通道特征融合的视觉任务，如视频分析和医学图像处理。
'''
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class CGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
if __name__ == '__main__':
    models = CGLU(in_features=768, hidden_features=512, out_features=768)

    # 1.如何输入的是图片4维数据 . CV方向的小伙伴都可以拿去使用
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(2, 768, 14, 14)
    input = input_img
    input_img = input_img.reshape(2, 768, -1).transpose(-1, -2)
    # 运行前向传递
    output = models(input_img,14,14)
    output = output.view(2, 768, 14, 14)  # 将三维度转化成图片四维度张量
    # 输出输入图片张量和输出图片张量的形状
    print("CV_CGLU_input size:", input.size())
    print("CV_CGLU_Output size:", output.size())

    # 2.如何输入的3维数据 . NLP方向的小伙伴都可以拿去使用
    B, N, C = 2, 196, 768  # 批量大小、序列长度、特征维度
    H, W = 14, 14  # 重塑后的高度和宽度
    input = torch.randn(B, N, C)
    output = models(input,H,W)
    print('NLP_CGLU_size:',input.size())
    print('NLP_CGLU_size:',output.size())

