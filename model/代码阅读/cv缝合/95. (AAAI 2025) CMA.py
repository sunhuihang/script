import torch
from torch import nn
import torch.nn.functional as F

# 交叉模态注意力模块
class CrossModelAtt(nn.Module):
    def __init__(self, feature_dim, height, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的参数
        self.softmax = nn.Softmax(dim=-1)  # Softmax 用于归一化注意力权重

    def forward(self, img_feat, text_feat):
        # img_feat和text_feat是输入的图像和文本特征图
        # img_feat和text_feat的形状为 [B, C, H, W]，其中B是批量大小，C是通道数，H和W是特征图的高度和宽度
        B, C, H, W = img_feat.shape

        # 1: 特征图展平
        q = img_feat.view(B, C, -1)  # [B, C, H*W]
        k = text_feat.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        # 2: 计算注意力感知矩阵
        attention_map = torch.bmm(q, k)  # [B, C, C]
        attention_map = self.softmax(attention_map)  # [B, C, C]

        # 3: 融合
        v = text_feat.view(B, C, -1)  # [B, C, H*W]
        attention_info = torch.bmm(attention_map, v)  # [B, C, H*W]

        # 重构为原始的H和W维度
        attention_info = attention_info.view(B, C, H, W)

        # 加权和原特征图
        output = self.gamma * attention_info + img_feat  # 加权融合后的结果

        return output

# 主函数进行张量测试
if __name__ == "__main__":
    batch_size = 1
    channels = 3
    height, width = 256, 256

    # 创建图像和文本特征图张量
    img_feat = torch.randn(batch_size, channels, height, width).cuda()  # 输入图像特征图
    text_feat = torch.randn(batch_size, channels, height, width).cuda()  # 输入文本特征图

    # 初始化交叉模态注意力层
    cross_attention = CrossModelAtt(feature_dim=channels, height=height, width=width).cuda()
    print(cross_attention)
    print("\n哔哩哔哩CV缝合救星!\n")

    # 前向传播测试
    output = cross_attention(img_feat, text_feat)

    # 打印输入和输出的形状
    print(f"Input image feature shape:  {img_feat.shape}")
    print(f"Input text feature shape :  {text_feat.shape}")
    print(f"Output shape             :  {output.shape}")
