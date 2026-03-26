import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
"""
CV缝合救星魔改创新7：空间注意力的动态调整 (Dynamic Adjustment of Spatial Attention)
一、背景
传统的空间注意力图是通过固定的操作生成的，这种生成方式无法根据输入特征的变化灵活调整空间注意力图。
例如，在某些输入情况下，某些区域可能对模型更为重要，而固定的空间注意力图无法灵活响应这些变化。因
此，现有模型缺乏在不同输入特征下进行自适应调整的能力。

二、改进方法
1. 通过条件卷积（如利用输入的特征来生成卷积核）来生成动态的空间注意力图。卷积网络能够有效地学习
到特征图中不同区域的重要性，并根据每个批次的输入灵活地调整注意力图。
2. 设计了一个由两个卷积层构成的网络，其中第一个卷积层提取特征，第二个卷积层输出注意力图。该网络
使用Sigmoid激活函数将注意力图限制在[0, 1]区间内，确保注意力值是可解释和可用的。

三、改进后的优势
1. 自适应特征调整：与固定的空间注意力图不同，动态空间注意力图根据输入特征的变化进行自适应调整，提高
了模型的表现力。
2. 增强模型灵活性：能够根据不同的输入，动态生成适合该输入的注意力图，从而优化模型对特征的关注度。
3. 提升模型性能：通过动态调整，模型能够更有效地学习和关注重要的区域，从而提升了对复杂场景的适应能力。
"""



class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(DynamicSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入：x (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()

        # 通过条件卷积生成空间注意力图
        attention_map = self.conv1(x)
        attention_map = self.conv2(F.relu(attention_map))

        # 将输出映射到 [0, 1] 之间
        attention_map = self.sigmoid(attention_map)

        # 将注意力图应用于输入特征
        out = x * attention_map  # (B, C, H, W)
        return out


# 测试代码
if __name__ == "__main__":
    # 假设输入特征图大小为 (batch_size=8, in_channels=64, height=32, width=32)
    input_tensor = torch.randn(8, 64, 32, 32).cuda()
    print(input_tensor.shape)

    # 初始化动态空间注意力模块
    dynamic_attention = DynamicSpatialAttention(64).cuda()

    # 生成动态调整后的输出
    output_tensor = dynamic_attention(input_tensor)

    print(output_tensor.size())  # 输出: torch.Size([8, 64, 32, 32])
