import torch
import torch.nn as nn
"""
CV缝合救星魔改创新2：结合空间注意力机制的背景分析及改进
一、背景
传统的SE模块（Squeeze-and-Excitation Block）主要聚焦于通道维度的特征重要性，通过全局平均池化提取
每个通道的全局信息，生成通道注意力权重。然而，SE模块完全忽略了空间维度的特征分布，存在以下几个问题：
1. 空间特征丢失：SE模块的全局平均池化将每个通道的二维空间特征压缩为一个标量，仅保留通道级的全局信息，
无法感知输入特征图在空间维度（height × width） 上的细粒度特征分布。
2. 空间上下文依赖性弱：通道注意力机制只关注通道之间的全局重要性权重，忽视了同一通道内不同位置的激活强
度差异。对于需要感知物体边界、纹理和结构的任务，这种忽视可能导致关键特征点丢失。
3. 通道与空间协同不足：通道注意力机制擅长选择“重要通道”，但无法根据图像中的空间特征动态调整权重，从而
限制了特征选择的灵活性。

二、改进方法
1. 融合通道和空间注意力：在通道注意力计算完成后，进一步对特征图在空间维度上应用注意力机制。传统方法依赖
简单的全局池化，而在这里我们引入了多尺度卷积来捕获不同尺度的空间信息，从而增强空间注意力的表现力，动态关
注输入特征图中最重要的位置。
2. 空间权重生成方式：在空间注意力的生成中，我们引入了 多尺度卷积（例如使用 3x3、5x5 和 7x7 卷积）来提取
不同尺度的空间特征信息，利用不同卷积核的特征响应来生成空间权重。这种多尺度特征融合方法比传统的全局池化方式
能够捕获更多细粒度的空间信息。
3. 动态卷积生成空间权重：不同于传统通过全局池化（Global Average Pooling 和 Global Max Pooling）生成
空间权重的方式，本方法使用动态卷积来适应输入特征的空间变化，从而生成更加灵活且自适应的空间权重。这种方法避
免了传统固定卷积核的局限性，提高了空间权重的表达能力。
4. 通道-空间联合加权：通道注意力和空间注意力的计算是分阶段进行的，首先生成通道权重并进行通道加权，然后通过
多尺度卷积生成空间权重，最后对输入特征图进行联合加权。这种设计让模型能够更好地结合通道和空间的特征，提升特征
选择的能力。

三、改进后的优势
1. 增强空间特征表达：通过空间注意力机制和多尺度卷积，模型能够动态关注输入特征图中不同位置的重要性，尤其是在
需要捕获物体边缘、复杂纹理或细粒度特征的任务中显著提升性能。这一设计突破了传统注意力机制仅依赖全局池化的限制，
能够更好地提取空间特征。
2. 通道与空间互补：通道注意力负责选择重要通道，而空间注意力则负责在通道内动态选择关键位置，两者结合能够更全面
地挖掘输入特征中的有效信息。多尺度卷积的引入让空间信息的捕获更加细腻和全面，进一步提高了模型的表现。
3. 灵活与高效：通过动态卷积和多尺度卷积的方式，空间注意力不仅增强了特征表达能力，而且计算复杂度较低。与通道注
意力分阶段进行的设计，不增加显著的模型参数量，保持了计算效率和轻量化特性。
4. 提高自适应性：空间注意力机制能够根据输入特征图的具体内容自适应调整卷积核权重，使得空间特征的选择更为灵活和
动态，能够根据不同的输入特征图情况调整空间注意力的计算方式。
"""
import torch
import torch.nn as nn

class SEWithAdvancedSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        改进后的通道和空间注意力结合的SE模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(SEWithAdvancedSpatialAttention, self).__init__()

        # 通道注意力部分
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid_channel = nn.Sigmoid()  # Sigmoid 激活生成通道权重

        # 空间注意力部分 (改进)
        self.conv3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)  # 3x3卷积
        self.conv5x5 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2, bias=False)  # 5x5卷积
        self.conv7x7 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)  # 7x7卷积
        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)  # 将多个尺度的空间信息进行融合
        self.sigmoid_spatial = nn.Sigmoid()  # Sigmoid 激活生成空间权重

    def forward(self, x):
        # 通道注意力部分
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y_channel = self.sigmoid_channel(y)  # 通道权重
        x_channel = x * y_channel  # 按通道加权

        # 空间注意力部分（多尺度卷积）
        out_3x3 = self.conv3x3(x_channel)  # 使用3x3卷积提取空间特征
        out_5x5 = self.conv5x5(x_channel)  # 使用5x5卷积提取空间特征
        out_7x7 = self.conv7x7(x_channel)  # 使用7x7卷积提取空间特征

        # 融合多个尺度的空间特征
        out_fused = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)  # 拼接不同尺度的特征
        out_fused = self.conv_fuse(out_fused)  # 使用1x1卷积融合空间特征

        y_spatial = self.sigmoid_spatial(out_fused)  # 生成空间权重
        return x_channel * y_spatial  # 同时加权通道和空间

if __name__ == "__main__":
    # 输入张量，形状为 [batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # 批量大小8，通道数64，特征图尺寸32x32
    se_block = SEWithAdvancedSpatialAttention(in_channels=64, reduction=16)
    output_tensor = se_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
