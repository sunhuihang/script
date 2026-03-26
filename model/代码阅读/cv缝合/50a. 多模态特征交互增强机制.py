import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3
"""
CV缝合救星魔改创新1：多模态特征交互增强机制
在原有的 FFM 模块中，虽然实现了 CNN 和 Transformer 特征的融合，但对于不同模态特征之间的交互方式还可以进一步优化。
新的创新点是引入一种多模态特征交互增强机制，在跨域融合块（CFB）和相关性增强操作中，增加一个基于特征图空间和通道维度
的交叉模态注意力模块。这个模块能够更精细地捕捉 CNN 特征和 Transformer 特征在不同位置和通道上的关联关系，从而增强
特征融合的效果。
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


class CrossModalAttention(nn.Module):
    def __init__(self, channels):
        super(CrossModalAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, cnn_feature, transformer_feature):
        m_batchsize, C, height, width = cnn_feature.size()
        proj_query = self.query_conv(cnn_feature).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(transformer_feature).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(transformer_feature).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + cnn_feature
        return out


class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),  # 修改为 Conv2d
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),  # 修改为 Conv2d
            nn.Conv2d(f, channels, 3, padding=1),  # 修改为 Conv2d
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


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv1x1_cnn = nn.Conv2d(in_channels[0], in_channels[1], 1)
        self.conv1x1_transformer = nn.Conv2d(in_channels[0], in_channels[1], 1)
        self.ca = channel_att(in_channels[1])
        self.cfb = CrossModalAttention(in_channels[1])
        self.local_attention_cnn = LocalAttention(in_channels[1])  # 添加 LocalAttention 到 CNN 特征
        self.local_attention_transformer = LocalAttention(in_channels[1])  # 添加 LocalAttention 到 Transformer 特征
        self.ffb = nn.Sequential(
            C3(in_channels[1] * 3, in_channels[1]),  # 将输入通道数修改为拼接后的实际通道数
            nn.Conv2d(in_channels[1], in_channels[1], 3, padding=1)
        )

    def forward(self, cnn_feature, transformer_feature):
        # 特征预处理与对齐
        cnn_feature_aligned = self.conv1x1_cnn(cnn_feature)
        transformer_feature_aligned = self.conv1x1_transformer(transformer_feature)

        # 局部注意力增强
        cnn_feature_aligned = self.local_attention_cnn(cnn_feature_aligned)
        transformer_feature_aligned = self.local_attention_transformer(transformer_feature_aligned)

        # 通道注意力调整
        cnn_feature_aligned = self.ca(cnn_feature_aligned)
        transformer_feature_aligned = self.ca(transformer_feature_aligned)

        # 跨域融合与增强
        cross_fused_feature = self.cfb(cnn_feature_aligned, transformer_feature_aligned)

        # 融合与输出
        fused_feature = torch.cat([cnn_feature_aligned, transformer_feature_aligned, cross_fused_feature], dim=1)
        output = self.ffb(fused_feature)

        return output



if __name__ == "__main__":
    cnn_input = torch.randn(1, 32, 64, 64)
    transformer_input = torch.randn(1, 32, 64, 64)
    ffm = FeatureFusionModule([32, 64])
    output = ffm(cnn_input, transformer_input)
    print('input_size:', cnn_input.size())
    print('output_size:', output.size())