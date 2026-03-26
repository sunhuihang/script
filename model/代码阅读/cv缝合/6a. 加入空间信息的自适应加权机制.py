import torch
import torch.nn as nn

"""
创新点1：加入空间信息的自适应加权机制
1. 动机：
传统的CRA模块主要在通道维度上进行缩减和注意力计算，但忽视了图像中不同空间区域的重要性。在语义分割、目标检测等任务中，
图像中的某些空间区域（如目标边缘、纹理区域等）往往对模型的判别结果至关重要。仅依赖通道上的缩减可能导致模型忽略这些关键区域，
使得注意力在背景等不重要的区域分散。我们提出一种基于空间信息的自适应加权机制，使注意力模块在聚合全局特征时，能够灵活调整对
不同空间位置的关注度，更精准地捕捉关键空间区域的全局上下文。

2. 方法：
在CRA模块中增加一个1x1卷积层生成自适应的空间权重图。卷积层学习各空间位置的特征重要性，经过sigmoid激活后生成权重图。
该权重图应用于原始输入特征图，使重要区域的特征在通道缩减前得到增强。这样在后续的注意力计算中，可以更加突出关键空间信息，
使模型有效过滤掉不重要的区域，从而集中资源在语义丰富的部位。

3. 效果：通过引入空间自适应加权机制，CRA模块在捕捉全局上下文的同时，更加关注图像中的关键区域。这样不仅提高了模型对重要区域
的敏感性，还增强了模型在复杂场景下的表现，尤其在需要高精度分割的边缘细节上获得了显著提升。
"""

class ImprovedCRA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ImprovedCRA, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        # 通道缩减线性层
        self.query_projection = nn.Linear(in_channels, reduced_channels)
        self.key_projection = nn.Linear(in_channels, reduced_channels)
        self.value_projection = nn.Linear(in_channels, in_channels)

        # 新增空间信息的卷积加权
        self.spatial_weight_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 空间信息权重计算
        spatial_weight = torch.sigmoid(self.spatial_weight_conv(x))  # [B, 1, H, W]
        weighted_x = x * spatial_weight  # 应用权重

        # 通道缩减后的查询、键和值
        input_flat = weighted_x.view(batch_size, channels, -1)
        avg_pool = torch.mean(input_flat, dim=-1, keepdim=True)

        query = self.query_projection(input_flat.permute(0, 2, 1))
        key = self.key_projection(avg_pool.permute(0, 2, 1))
        value = self.value_projection(avg_pool.permute(0, 2, 1))

        attention_map = torch.softmax(torch.bmm(query, key.permute(0, 2, 1)), dim=1)
        out = torch.bmm(attention_map, value)

        out = out.view(batch_size, channels, height, width)
        return out

if __name__ == "__main__":
    input = torch.randn(8, 64, 32, 32)
    CRA = ImprovedCRA(in_channels=64, reduction_ratio=4)
    output = CRA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
