import torch
import torch.nn as nn
"""
CV缝合救星-SE Conv
一、修改点说明：
1. 使用 nn.Conv2d 替代了原有的全连接层 nn.Linear，保持输入维度和输出维度一致。
2. 卷积核大小设为 1x1，作用等同于全连接层，能够在保持全局平均池化后张量形状的情况下
执行降维和恢复维度操作。
二、优势：
1. 使用卷积操作能更高效地在多通道特征图中应用权重计算。
2. 卷积可以直接作用于特征图，不需要展平操作（view），更符合图像处理的习惯。
"""
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        Squeeze-and-Excitation (SE) 模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例，用于降低通道维度
        """
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        # 使用卷积代替全连接层，调整通道数
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活函数
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，用于生成通道注意力权重

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入张量的批量大小和通道数
        # Squeeze操作：通过全局平均池化提取全局信息
        y = self.global_avg_pool(x)  # 输出形状：[批量, 通道, 1, 1]
        # Excitation操作：通过两层卷积计算注意力权重
        y = self.conv1(y)  # 降维，形状变为 [批量, 通道 // reduction, 1, 1]
        y = self.relu(y)  # ReLU 激活
        y = self.conv2(y)  # 恢复维度，形状变为 [批量, 通道, 1, 1]
        y = self.sigmoid(y)  # 生成注意力权重
        # 将注意力权重应用到输入张量
        return x * y  # 按通道加权输入特征

# Test the SEBlock
if __name__ == "__main__":
    # Example tensor with shape [batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # Batch size 8, 64 channels, 32x32 feature map
    se_block = SEBlock(in_channels=64, reduction=16)  # Create SE block
    output_tensor = se_block(input_tensor)  # Forward pass
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)