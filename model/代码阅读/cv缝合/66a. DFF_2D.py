import torch
import torch.nn as nn

# 动态特征融合（DFF）模块 - 2D 版本
# B站：CV缝合救星原创出品
class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 2D 自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 注意力卷积层，使用 2D 卷积
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 通道减少卷积层，使用 2D 卷积
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        # 两个 2D 卷积层用于计算注意力
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        # Sigmoid 激活函数
        self.nonlin = nn.Sigmoid()
        # B站：CV缝合救星原创出品

    def forward(self, x, skip):
        # 沿着通道维度拼接输入特征
        output = torch.cat([x, skip], dim=1)
        # 计算注意力权重
        att = self.conv_atten(self.avg_pool(output))
        # 应用注意力权重
        output = output * att
        # 减少通道数量
        output = self.conv_redu(output)
        # B站：CV缝合救星原创出品
        # 计算另一个注意力权重
        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        # 应用另一个注意力权重
        output = output * att
        return output

if __name__ == '__main__':
    # 生成随机输入数据，2D 图像维度 (B, C, H, W)
    input1 = torch.randn(3, 32, 64, 64)
    input2 = torch.randn(3, 32, 64, 64)
    # 初始化 DFF 模块
    model = DFF(32)
    # 前向传播
    output = model(input1, input2)
    print("DFF_input size:", input1.size())
    print("DFF_Output size:", output.size())