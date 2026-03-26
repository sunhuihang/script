import torch
import torch.nn as nn
"""
CV缝合救星魔改创新6：引入归一化层
一、背景和动机
在卷积神经网络中，通常使用批归一化（Batch Normalization, BN）来提高训练的稳定性和加速收敛。然而，批归一化存在一些限制，
尤其在小批量（小batch size）或递归神经网络（RNNs）中效果不佳。在这种情况下，层归一化（Layer Normalization, LN）成为
一个有效的替代方案。
1. 批归一化（BN） 依赖于批量内的统计信息（均值和方差），但它不适用于小batch size或时间序列数据的情况。
2. 层归一化（LN） 是对每个样本的所有特征维度进行归一化。与批归一化不同，层归一化的计算仅在单个样本内部进行，因此不依赖于批量大小。
3. 引入层归一化后，模型会在每个输入通道的层级上进行归一化，从而可能提升模型的泛化能力并加速收敛。
二、思路
在通道压缩（通过SE模块）之前，或者每次进行特征变换时加入层归一化。通过这种方式，层归一化可以帮助在压缩和激活过程中保持特征的稳定性
和尺度一致性。
"""

class LayerNormSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        引入层归一化（Layer Normalization）的SE模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(LayerNormSEBlock, self).__init__()

        # LayerNorm用于标准化每个样本的所有特征
        self.layer_norm = nn.LayerNorm(in_channels)

        # SE模块的全连接层，用于学习通道的压缩比例
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)  # 降维
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)  # 恢复维度
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活生成注意力权重

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入的维度

        # 1. 应用层归一化
        # 先展平空间维度 [b, c, h * w]，然后在通道维度上进行归一化
        x = x.view(b, c, -1)  # 将空间维度展平
        x = x.permute(0, 2, 1)  # 调整维度以使得LayerNorm应用于通道维度
        x = self.layer_norm(x)  # 在通道维度上进行归一化
        x = x.permute(0, 2, 1)  # 恢复通道和空间维度的顺序
        x = x.view(b, c, h, w)  # 恢复原始的空间形状

        # 2. 通过全局池化得到每个通道的全局信息
        avg_pool = self.global_avg_pool(x)  # 对每个通道进行全局平均池化
        avg_pool = avg_pool.view(b, c)  # 展平以便输入到全连接层

        # 3. 计算通道的注意力权重
        y = self.fc1(avg_pool)  # 降维
        y = self.relu(y)
        y = self.fc2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成通道注意力权重

        # 4. 按通道加权输入特征
        y = y.view(b, c, 1, 1)  # 将注意力权重调整为 [b, c, 1, 1] 的形状
        output = x * y  # 按通道加权输入特征

        return output


# 测试代码
if __name__ == "__main__":
    # 输入示例张量：[batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # Batch size 8, 64 channels, 32x32 feature map
    layer_norm_se_block = LayerNormSEBlock(in_channels=64, reduction=16)  # 创建层归一化SE模块
    output_tensor = layer_norm_se_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)


