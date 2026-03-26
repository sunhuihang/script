import torch
import torch.nn as nn
# 代码： https://github.com/hyunwoo137/MetaSeg/tree/main?tab=readme-ov-file
# 论文：https://openaccess.thecvf.com/content/WACV2024/papers/Kang_MetaSeg_MetaFormer-Based_Global_Contexts-Aware_Network_for_Efficient_Semantic_Segmentation_WACV_2024_paper.pdf

'''
MetaSeg：基于 MetaFormer的全局上下文感知网络用于高效的语义分割    WACV 2024 顶会
通道缩减注意力即插即用模块：CRAttention

背景：近期的分割方法表明，采用基于CNN的骨干网络来提取空间信息，再通过解码器提取全局信息，
比使用基于Transformer的骨干网络配合CNN解码器的效果更佳。这一发现促使我们选择使用
包含MetaFormer模块的CNN骨干网络，并设计了基于MetaFormer的解码器，其中集成了一个
新颖的自注意力模块，专用于捕获全局上下文信息。

模块：CRA模块是为了解决语义分割任务中自注意力机制的计算效率问题而提出的。传统的自注意力机制
在捕捉全局上下文时计算量较大，特别是在高分辨率特征的情况下计算成本更高。CRA通过在语义
分割和医学图像分割任务中，提升计算效率和准确性来优化自注意力应用。

适用任务：
1.图像分类
2.语义分割：如ADE20K、Cityscapes等数据集。
3.实例分割与目标检测：用于提高分割和检测任务中的边界识别精度。
4.如Synapse数据集，通过捕获全局上下文提升医学图像分割的准确性。
5.适用于各种需要注意力机制的图像处理任务。
'''
class CRA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CRA, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        self.query_projection = nn.Linear(in_channels, reduced_channels)
        self.key_projection = nn.Linear(in_channels, reduced_channels)
        self.value_projection = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        input_flat = x.view(batch_size, channels, -1)

        avg_pool = torch.mean(input_flat, dim=-1, keepdim=True)

        query = self.query_projection(input_flat.permute(0, 2, 1))
        key = self.key_projection(avg_pool.permute(0, 2, 1))
        value = self.value_projection(avg_pool.permute(0, 2, 1))

        attention_map = torch.softmax(torch.bmm(query, key.permute(0, 2, 1)), dim=1)
        out = torch.bmm(attention_map, value)

        out = out.view(batch_size, channels, height, width)  # 还原成原始形状
        return out

if __name__ == "__main__":
    input = torch.randn(8, 64, 32, 32)
    CRA = CRA(in_channels=64, reduction_ratio=4)
    output = CRA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
