import torch
import torch.nn as nn
"""
CV缝合救星魔改创新1：多头通道注意力机制背景分析及改进
一、背景
传统的SE模块（Squeeze-and-Excitation Block） 是通过全局平均池化对每一个通道的全局特征进行压缩，
生成全局语义信息，再通过两个全连接层对通道维度进行缩放和恢复，从而生成注意力权重。然而，传统的SE模
块在以下几个方面存在不足：
1. 单一通道压缩策略：SE模块仅使用全局平均池化提取通道全局特征，忽略了通道之间可能存在的细粒度交互
关系。简单的通道加权机制不足以充分捕获通道之间的复杂相关性。
2. 全通道统一操作：SE模块对所有通道执行统一的加权操作，未能充分利用通道的组内特性（如不同通道可能属
于不同的语义组）。这种处理方式容易丢失通道内的细粒度信息。
3. 计算资源限制：在高维特征图中，通道数往往较多，统一操作导致注意力计算成本较高，特别是在使用较深层
网络时。
二、改进方法
针对上述问题，引入多头通道注意力机制（Multi-Head Channel Attention），主要包括以下改进：
1. 通道分组与多头机制：将通道划分为多个子组，每个子组（即一个头）独立计算注意力。这样可以实现通道之间
的分组处理，捕获更精细的特征交互。
2. 局部特征增强：每个头通过独立的全局平均池化和注意力权重计算，可以更好地关注该组内的局部特征，同时避
免全局特征权重分布的不均衡问题。
3. 分组并行化：
多头机制通过分组并行计算注意力，既可以减少每次注意力计算的通道数，降低计算复杂度，又提升了特征选择的灵
活性。
三、改进后的优势
1. 捕获更多细粒度特征：多头机制通过划分通道组，使模型能够在局部范围内专注于细粒度特征交互关系，从而提升
模型的表达能力。
2. 提升计算效率：每个子组的计算维度较低，注意力计算的复杂度显著降低，在特征图较大时优势更加明显。
3. 模型鲁棒性增强：多头机制可以防止单一通道注意力的偏倚问题，使模型在不同场景下具有更好的泛化能力。
"""
class MultiHeadSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, num_heads=4):
        """
        多头通道注意力机制模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        :param num_heads: 注意力头的数量
        """
        super(MultiHeadSEBlock, self).__init__()
        assert in_channels % num_heads == 0, "通道数必须能被注意力头数量整除"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # 每个头的通道数
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1 = nn.Conv2d(self.head_dim, self.head_dim // reduction, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(self.head_dim // reduction, self.head_dim, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活生成权重

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入维度
        # 分割通道为多个头
        x_split = x.view(b * self.num_heads, self.head_dim, h, w)  # 将通道分为多个头，每个头作为一个批次
        # 对每个头进行操作
        y = self.global_avg_pool(x_split)  # 全局平均池化，形状 [b*num_heads, head_dim, 1, 1]
        y = self.conv1(y)  # 降维
        y = self.relu(y)
        y = self.conv2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成注意力权重
        # 将注意力权重应用到输入张量
        x_split = x_split * y  # 对每个头的特征加权
        # 恢复形状
        x_out = x_split.view(b, c, h, w)  # 合并所有头
        return x_out

# 测试 MultiHeadSEBlock
if __name__ == "__main__":
    # 输入张量，形状为 [batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # 批量大小8，通道数64，特征图尺寸32x32
    se_block = MultiHeadSEBlock(in_channels=64, reduction=16, num_heads=4)  # 实例化多头SE模块
    output_tensor = se_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
