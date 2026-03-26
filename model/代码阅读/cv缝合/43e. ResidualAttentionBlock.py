import torch
import torch.nn as nn
"""
CV缝合救星魔改创新4：注意力残差机制（Residual Attention Mechanism）
一、背景与问题分析
传统的SE模块（Squeeze-and-Excitation Block） 已经在一定程度上改进了通道注意力的机制，但在许多情况下，
SE模块可能会导致信息的丢失或梯度消失问题。原因是SE模块通过全局池化来对输入特征进行压缩和重标定，而这种压
缩可能会导致一些细节信息的丢失，尤其在深度网络中，特征图会不断减少维度，这些细节信息对于任务的表现至关重要。
此外，传统的SE模块是基于全局池化产生的注意力权重来进行调整的，这种做法虽然能增强通道之间的相关性，但缺乏对
残差学习的支持。在一些深度神经网络中，加入残差连接可以有效缓解梯度消失，并提高信息的流动性。
问题：
1. 传统的SE模块无法完全解决信息丢失的问题。
2. 没有充分利用残差结构来避免梯度消失，提升网络的深度可训练性。

二、改进方法
注意力残差机制（Residual Attention Mechanism），该机制的核心思想是结合残差学习与注意力机制，通过将SE模
块与残差连接相结合，来增强信息流动，减少信息丢失，并改进注意力机制的效果。
实现思路：
1. 残差连接：将SE模块输出的注意力权重与输入特征相加，形成残差连接。这样可以有效避免梯度消失，同时保留了输入的
原始信息。
2. SE模块集成：在残差连接中引入SE模块的通道注意力机制，进行特征重标定，从而增强对有用信息的关注。

三、改进后的优势
1. 信息流动增强：通过残差连接，信息能够更好地流动到网络的深层，减少梯度消失的风险。
2. 提高训练稳定性：残差连接可以帮助训练更深的网络，同时减少信息的丢失。
3. 增强的通道注意力机制：SE模块通过注意力机制增强了对特征图中重要通道的关注，结合残差连接后，进一步提升了模型
性能。
4. 更好的收敛性：残差连接能够加速网络的训练和收敛，特别是在深层网络中。
"""

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        注意力残差机制模块，结合SE模块与残差连接。
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(ResidualAttentionBlock, self).__init__()

        # SE模块部分
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活生成权重

    def forward(self, x):
        # SE模块的注意力计算
        avg_pool = self.global_avg_pool(x)  # 全局平均池化
        y = self.conv1(avg_pool)  # 降维
        y = self.relu(y)
        y = self.conv2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成注意力权重

        # 注意力残差机制
        attention_output = x * y  # 按通道加权输入特征
        residual_output = x + attention_output  # 加入残差连接

        return residual_output  # 输出加了残差连接的注意力输出


# 测试代码
if __name__ == "__main__":
    # 输入示例张量：[batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # Batch size 8, 64 channels, 32x32 feature map
    res_att_block = ResidualAttentionBlock(in_channels=64, reduction=16)  # 创建残差注意力块
    output_tensor = res_att_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
