import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        Squeeze-and-Excitation (SE) 模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例，用于降低通道维度
        """
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)  # 第一个全连接层，降低通道维度
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活函数
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)  # 第二个全连接层，恢复通道维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，用于生成通道注意力权重

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入张量的批量大小和通道数
        # Squeeze操作：通过全局平均池化提取全局信息
        y = self.global_avg_pool(x).view(b, c)  # 输出形状：[批量, 通道]
        # Excitation操作：通过两个全连接层计算注意力权重
        y = self.fc1(y)  # 降维
        y = self.relu(y)  # ReLU 激活
        y = self.fc2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成注意力权重
        y = y.view(b, c, 1, 1)  # 调整维度以便与输入张量相乘
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
