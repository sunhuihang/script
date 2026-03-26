import torch
import torch.nn as nn

"""
创新2：注意力融合机制
一、背景
传统的空间注意力生成方法通常使用平均池化和最大池化来提取信息，但这种方式可能会丢失一些细粒度的空间信息。通过将多种注意力生成方法结合起来，
可以使模型更全面地学习空间重要性。
二、实现方法
通过融合不同的池化方式（如最大池化、平均池化和L2池化），生成多个注意力图，并将它们加权融合，从而得到更加准确的空间注意力图。
"""

class AttentionFusionSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        """
        使用注意力融合机制生成空间注意力图
        :param in_channels: 输入的通道数
        """
        super(AttentionFusionSpatialAttention, self).__init__()

        # 全局池化，生成相同大小的池化图
        self.avg_pool = nn.AdaptiveAvgPool2d((32, 32))  # 输出大小设置为32x32，匹配输入特征图
        self.max_pool = nn.AdaptiveMaxPool2d((32, 32))  # 输出大小设置为32x32，匹配输入特征图
        self.l2_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # 用均值池化模拟L2池化

        self.conv = nn.Conv2d(in_channels * 3, 1, kernel_size=1, bias=False)  # 融合多个池化的输出

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        生成融合后的空间注意力图，并加权输入特征
        :param x: 输入特征图
        :return: 加权后的输出特征图
        """
        # 获取不同池化方式的输出，确保池化后的大小与输入特征图一致
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        l2_pool = self.l2_pool(x)

        # 拼接多个池化输出，得到一个多维特征图
        fused_attention = torch.cat([avg_pool, max_pool, l2_pool], dim=1)

        # 通过卷积融合所有池化方式的特征
        attention_map = self.conv(fused_attention)
        attention_map = self.sigmoid(attention_map)

        # 按照注意力图加权输入特征
        output = x * attention_map
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
    attention_fusion_attention = AttentionFusionSpatialAttention(in_channels=64)
    output_tensor = attention_fusion_attention(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
