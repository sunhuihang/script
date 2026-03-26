import torch
import torch.nn as nn
"""
CV缝合救星魔改创新3：引入多尺度池化的背景分析及改进
一、背景
传统的SE模块（Squeeze-and-Excitation Block）使用全局平均池化（Global Average Pooling, GAP）
提取每个通道的全局语义信息。虽然这种方式有效，但它存在以下局限性：
1. 单尺度特征限制：全局平均池化压缩特征图到单个值，导致模型只关注全局特征而忽略不同尺度的细粒度信息。
这在需要处理多尺度特征（如目标检测中不同大小的目标）时表现不足。
2. 缺乏上下文信息整合：单一尺度的全局特征对上下文的表达能力有限，尤其是当目标的局部特征对任务重要性较
高时（如小目标检测），单尺度池化可能无法捕获足够的信息。
3. 信息丢失问题：平均池化是一种全局的统计运算，可能忽略局部激活的重要性，难以保留特征图中关键区域的细节
信息。
二、改进方法
针对上述问题，引入多尺度池化（Multi-Scale Pooling, MSP） 方法对SE模块进行改进，主要思路如下：
1. 全局与局部特征结合：同时使用多种尺度的池化方法（如全局平均池化和全局最大池化）来提取特征，增强对不同上
下文特征的表达能力。
2. 多尺度特征融合：将多种池化结果融合（如通过加和或拼接）以生成更全面的通道描述，保留全局和局部激活的信息。
3. 改进注意力权重生成：在融合后的多尺度特征基础上，计算通道权重，使模型能够更灵活地关注全局和局部的重要区域。
"""
class MultiScaleSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        引入多尺度池化的SE模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(MultiScaleSEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活生成权重

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入维度
        # 多尺度池化
        avg_pool = self.global_avg_pool(x)  # 全局平均池化
        max_pool = self.global_max_pool(x)  # 全局最大池化
        # 融合池化特征（加和方式）
        y = avg_pool + max_pool  # 将平均池化和最大池化结果相加(可以继续缝合门控，通道拼接)
        y = self.conv1(y)  # 降维
        y = self.relu(y)
        y = self.conv2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成注意力权重
        return x * y  # 加权输入特征

if __name__ == "__main__":
    # 输入张量，形状为 [batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # 批量大小8，通道数64，特征图尺寸32x32
    se_block = MultiScaleSEBlock(in_channels=64, reduction=16)
    output_tensor = se_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)