import torch
import torch.nn as nn
import torch.nn.functional as F
"""
魔改创新9. 局部与全局特征融合空间注意力
背景说明：
在传统的空间注意力机制中，通过某种方式生成固定的注意力图，但这种方法无法根据输入的特征自适应调整关注区域。
为了弥补这个缺陷，局部与全局特征融合的空间注意力机制能够结合全局和局部信息，根据输入的特征动态地调整空间
注意力图。全局信息能够捕捉整体上下文，局部信息则关注细节特征，二者结合可以提升模型的表现力和适应性。
"""

class SpatialAttentionWithFusion(nn.Module):
    """
    局部与全局特征融合空间注意力模块
    该模块结合了局部和全局信息，通过全局平均池化提取全局特征，
    并通过卷积提取局部特征，再将两者融合生成空间注意力图。
    """

    def __init__(self, input_channels, kernel_size=3):
        super(SpatialAttentionWithFusion, self).__init__()

        # 全局平均池化，用于捕获全局特征
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 局部卷积，用于提取局部特征
        self.local_conv = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        # 最终生成注意力图的卷积层
        self.attention_conv = nn.Conv2d(input_channels * 2, 1, kernel_size=1)

        # 使用Sigmoid激活函数进行归一化
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: 输入的张量，形状为 (batch_size, channels, height, width)
        :return: 加权后的输出张量
        """
        # 全局特征
        global_features = self.global_avg_pool(input_tensor)  # 形状为 (batch_size, channels, 1, 1)

        # 局部特征
        local_features = self.local_conv(input_tensor)  # 形状为 (batch_size, channels, height, width)

        # 将全局特征广播到空间维度与输入张量相同
        global_features = global_features.expand(-1, -1, input_tensor.size(2),
                                                 input_tensor.size(3))  # (batch_size, channels, height, width)

        # 将全局与局部特征进行拼接
        fused_features = torch.cat([local_features, global_features],
                                   dim=1)  # 拼接后形状为 (batch_size, 2*channels, height, width)

        # 通过卷积层生成最终的空间注意力图
        attention_map = self.attention_conv(fused_features)  # 形状为 (batch_size, 1, height, width)
        attention_map = self.sigmoid(attention_map)  # 使用Sigmoid进行归一化

        # 输出加权后的输入张量
        output_tensor = input_tensor * attention_map  # 逐元素相乘

        return output_tensor

if __name__ == "__main__":
    # 假设输入特征图大小为 (batch_size=8, in_channels=64, height=32, width=32)
    input_tensor = torch.randn(8, 64, 32, 32).cuda()
    print(input_tensor.shape)

    # 初始化局部与全局特征融合空间注意力模块
    spatial_attention_fusion = SpatialAttentionWithFusion(64).cuda()

    # 生成加权后的输出
    output_fusion = spatial_attention_fusion(input_tensor)
    print("Fusion Attention Output Size:", output_fusion.size())

