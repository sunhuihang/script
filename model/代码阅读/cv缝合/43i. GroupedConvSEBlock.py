import torch
import torch.nn as nn
"""
CV缝合救星魔改创新8：引入分组卷积
一、背景
传统卷积操作对每个通道都进行计算，但这会导致较高的计算成本，尤其在输入通道数较多时。分组卷积（Grouped Convolution）
通过将输入通道划分为多个组，组内通道独立进行卷积计算，从而显著减少计算量。
二、实现方法
在SE模块中引入分组卷积，代替普通卷积，进一步优化计算效率。
"""
class GroupedConvSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, groups=4):
        """
        使用分组卷积的SE模块。
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        :param groups: 分组数量
        """
        super(GroupedConvSEBlock, self).__init__()

        # 分组卷积代替全连接操作
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, groups=groups, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, groups=groups, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # 通道注意力机制，使用分组卷积
        avg_pool = self.global_avg_pool(x)  # 全局平均池化
        y = self.conv1(avg_pool)  # 分组降维
        y = self.relu(y)
        y = self.conv2(y)  # 分组恢复维度
        y = self.sigmoid(y)  # 通道注意力权重

        # 按通道加权输入特征
        output = x * y
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)
    grouped_conv_se_block = GroupedConvSEBlock(in_channels=64, reduction=16, groups=4)
    output_tensor = grouped_conv_se_block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
