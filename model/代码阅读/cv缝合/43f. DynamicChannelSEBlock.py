import torch
import torch.nn as nn
"""

CV缝合救星魔改创新5：动态通道压缩比例（Dynamic Channel Reduction Ratio）
一、背景与问题分析
在 Squeeze-and-Excitation (SE) 模块中，我们通过全局池化计算通道的全局信息，然后使用一个固定的压缩比例（例如16）来减少通道数，
从而减小模型的复杂度。然而，固定的压缩比例并不适用于所有情况，因为每个特征图的通道信息在不同的输入下可能具有不同的重要性。过高的
压缩比例可能导致有用信息的丢失，而过低的压缩比例可能增加计算量并降低模型的表现。
问题：
1. 固定压缩比例可能导致不同任务或不同输入下的性能不稳定。
2. 静态压缩比例可能无法灵活地适应输入的多样性，限制了网络的适应性。

二、改进方法
动态通道压缩比例（Dynamic Channel Reduction Ratio），目的是根据输入特征图的特性自适应地选择压缩比例。这样，模型可以根据不同
的输入动态调整压缩比例，从而更有效地压缩重要信息，避免丢失有价值的特征。

三、实现思路：
1. 动态压缩比例：根据输入的特征图，使用一个小型的神经网络来动态预测一个压缩比例（例如，压缩比例在 [4, 32] 之间），根据该比例来
调整通道的压缩程度。
2. 与SE模块结合：在SE模块中，使用动态计算的压缩比例来替代固定的压缩比例，从而更灵活地控制模型的计算量。

四、改进后的优势
1. 更灵活的压缩：通过动态计算压缩比例，模型能够更好地适应不同输入的需求，避免固定压缩比例带来的性能瓶颈。
2. 提高特征表示能力：动态压缩比例能够帮助模型在保证计算效率的同时，充分保留输入特征的重要信息。
3. 提高模型适应性：在不同的任务或输入下，模型能够根据需要自适应调整压缩比例，从而在各种条件下都能取得较好的性能。
"""

import torch
import torch.nn as nn

class FullyDynamicSEBlock(nn.Module):
    def __init__(self, in_channels, reduction_min=4, reduction_max=32):
        """
        完全动态通道压缩比例的SE模块。
        :param in_channels: 输入的通道数
        :param reduction_min: 最小压缩比例
        :param reduction_max: 最大压缩比例
        """
        super(FullyDynamicSEBlock, self).__init__()

        # 使用一个小型神经网络来预测每个通道的压缩比例
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)  # 输出每个通道的压缩比例

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)  # 维度变换
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活生成权重

        self.reduction_min = reduction_min  # 最小压缩比例
        self.reduction_max = reduction_max  # 最大压缩比例

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入的维度

        # 通过全局池化得到每个通道的全局信息
        avg_pool = self.global_avg_pool(x)  # 全局平均池化
        avg_pool = avg_pool.view(b, c)  # 展平以便输入到全连接层

        # 使用全连接层预测每个通道的压缩比例
        reduction_ratios = self.fc1(avg_pool)
        reduction_ratios = self.fc2(reduction_ratios)  # 获取每个通道的压缩比例
        reduction_ratios = torch.sigmoid(reduction_ratios)  # 使用sigmoid将其归一化到[0, 1]范围

        # 将压缩比例映射到 [reduction_min, reduction_max] 范围内
        reduction_ratios = reduction_ratios * (self.reduction_max - self.reduction_min) + self.reduction_min

        # 动态计算每个通道的压缩比例
        reduction_ratios = reduction_ratios.view(b, c, 1, 1)  # 调整为通道级别的压缩比例

        # 使用动态压缩比例进行通道压缩
        y = self.conv1(x)  # 对特征进行卷积变换
        y = self.relu(y)
        y = self.conv2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成注意力权重

        # 按照每个通道的动态压缩比例加权输入特征
        output = x * y * reduction_ratios  # 加权输入特征
        return output


# 测试代码
if __name__ == "__main__":
    # 输入示例张量：[batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # Batch size 8, 64 channels, 32x32 feature map
    dynamic_se_block = FullyDynamicSEBlock(in_channels=64, reduction_min=4, reduction_max=32)  # 创建完全动态SE模块
    output_tensor = dynamic_se_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)


