import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.enabled = False
"""
CV缝合救星魔改创新10： 增强可解释性的空间注意力
背景说明：
空间注意力机制的核心思想是通过生成一个权重矩阵来强调输入特征图中重要区域的重要性。增强可解释性的空间注意力
不仅能提供准确的区域加权，还能帮助我们更清楚地理解模型为什么关注某些区域。通过卷积操作，我们生成一个空间注
意力图，并使用Sigmoid函数归一化到[0, 1]范围，从而确保权重值既合理又可解释。
"""

class ExplainableSpatialAttention(nn.Module):
    """
    增强可解释性的空间注意力模块
    该模块通过两个卷积层生成空间注意力图，并通过Sigmoid激活函数保证输出值在[0, 1]之间。
    可以帮助更好地理解模型为何关注某些区域。
    """

    def __init__(self, input_channels, kernel_size=3):
        super(ExplainableSpatialAttention, self).__init__()

        # 第一个卷积层，用于生成特征图
        self.conv1 = nn.Conv2d(input_channels, input_channels // 8, kernel_size=kernel_size, padding=kernel_size // 2)

        # 第二个卷积层，用于生成空间注意力图
        self.conv2 = nn.Conv2d(input_channels // 8, 1, kernel_size=3, padding=1)  # 输出空间注意力图

        # Sigmoid激活函数，用于将注意力图归一化到[0, 1]之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: 输入的张量，形状为 (batch_size, channels, height, width)
        :return: 加权后的输出张量，以及空间注意力图
        """
        # 首先通过卷积提取特征
        feature_map = self.conv1(input_tensor)  # 形状为 (batch_size, channels, height, width)

        # 通过第二个卷积生成空间注意力图
        attention_map = self.conv2(F.relu(feature_map))  # 形状为 (batch_size, 1, height, width)

        # 使用Sigmoid将注意力图归一化到[0, 1]范围
        attention_map = self.sigmoid(attention_map)

        # 输出加权后的输入张量
        output_tensor = input_tensor * attention_map  # 逐元素相乘

        return output_tensor, attention_map

if __name__ == "__main__":
    # 假设输入特征图大小为 (batch_size=8, in_channels=64, height=32, width=32)
    input_tensor = torch.randn(8, 64, 32, 32).cuda()
    print(input_tensor.shape)

    # 初始化增强可解释性的空间注意力模块
    explainable_attention = ExplainableSpatialAttention(64).cuda()

    # 生成加权后的输出和注意力图
    output_explainable, attention_map = explainable_attention(input_tensor)
    print("Explainable Attention Output Size:", output_explainable.size())
    print("Attention Map Size:", attention_map.size())

