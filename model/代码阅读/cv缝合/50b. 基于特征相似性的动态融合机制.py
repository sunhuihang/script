import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新2：基于特征相似性的动态融合机制
在特征融合过程中，通过计算 CNN 特征和 Transformer 特征在空间和通道维度上的相似性，
动态地调整融合权重。对于相似性高的区域，给予更高的融合权重，使得融合过程更加智能和自适应。
"""


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g  # (w + g)  # self.gate(x, w)


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SimilarityBasedFusion(nn.Module):
    def __init__(self, in_channels):
        super(SimilarityBasedFusion, self).__init__()
        self.conv1x1_cnn = nn.Conv2d(in_channels[0], in_channels[1], 1)
        self.conv1x1_transformer = nn.Conv2d(in_channels[0], in_channels[1], 1)
        self.ca_cnn = channel_att(in_channels[1])
        self.ca_transformer = channel_att(in_channels[1])
        self.similarity_conv = nn.Conv2d(in_channels[1], 1, 1)

    def forward(self, cnn_feature, transformer_feature):
        # 特征预处理与对齐
        cnn_feature_aligned = self.conv1x1_cnn(cnn_feature)
        transformer_feature_aligned = self.conv1x1_transformer(transformer_feature)

        # 通道注意力调整
        cnn_feature_aligned = self.ca_cnn(cnn_feature_aligned)
        transformer_feature_aligned = self.ca_transformer(transformer_feature_aligned)

        # 计算特征相似性
        similarity_map = self.similarity_conv(torch.abs(cnn_feature_aligned - transformer_feature_aligned))
        similarity_weight = torch.sigmoid(similarity_map)

        # 动态融合
        fused_feature = cnn_feature_aligned * similarity_weight + transformer_feature_aligned * (1 - similarity_weight)

        return fused_feature


if __name__ == "__main__":
    cnn_input = torch.randn(1, 32, 64, 64)
    transformer_input = torch.randn(1, 32, 64, 64)
    ffm = SimilarityBasedFusion([32, 64])
    output = ffm(cnn_input, transformer_input)
    print('input_size:', cnn_input.size())
    print('output_size:', output.size())