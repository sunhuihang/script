import torch
import torch.nn as nn

# 动态特征融合（DFF）模块 - 创新 2D 版本
# B站：CV缝合救星原创出品
"""
AMSA - FFM（Adaptive Multi - Scale Attention Feature Fusion Module，自适应多尺度注意力特征融合模块）
一. 创新点
1. 多尺度平均池化：除了原本的自适应平均池化*B站CV缝合救星*，引入多尺度的平均池化操作，以捕获不同尺度下的全局信息，
使模型能更好地适应不同
大小的特征模式。
2. 门控机制：在计算注意力权重时，加入门控机制*B站CV缝合救星*，通过额外的卷积层和 Sigmoid 函数来控制信息的流动，
增强模型对重要特征的关注能力。
3. 残差分支的融合：在通道减少后的特征图上添加一个残差分支，将原始输入特征进行一定的变换后与通道减少后的特征图相加，
有助于缓解梯度消失问题，保留更多原始特征信息*B站CV缝合救星*。

二. 代码解释
1. __init__ 方法：
A. 定义了三个不同尺度的自适应平均池化层 self.avg_pool1、self.avg_pool2 和 self.avg_pool3，用于捕获不同尺度的全局信息。
B. 为每个平均池化层对应的输出设置了独立的注意力卷积层 self.conv_atten1、self.conv_atten2 和 self.conv_atten3。
C. 引入了门控机制 self.gate_conv，通过一个 1x1 卷积层和 Sigmoid 函数来控制信息的流动。
D. 定义了通道减少卷积层 self.conv_redu 和残差分支的卷积层 self.residual_conv。
2. forward 方法：
A. *B站CV缝合救星*首先将输入特征 x 和 skip 沿着通道维度拼接。
B. 对拼接后的特征进行多尺度平均池化，并计算每个尺度下的注意力权重，然后将它们相加融合。
C. 通过门控机制对融合后的注意力权重进行调整。
D. 将调整后的注意力权重应用到拼接后的特征上*B站CV缝合救星*。
E. 进行通道减少操作，并通过残差分支将原始输入特征的变换结果与通道减少后的特征图相加。
F. 最后计算另一个注意力权重并应用到最终的特征图上。
"""
class InnovativeDFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 多尺度平均池化
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(4)

        # 注意力卷积层，使用 2D 卷积
        self.conv_atten1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_atten2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_atten3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 门控机制
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 通道减少卷积层，使用 2D 卷积
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # 残差分支
        self.residual_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # 两个 2D 卷积层用于计算注意力
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        # Sigmoid 激活函数
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        # 沿着通道维度拼接输入特征
        output = torch.cat([x, skip], dim=1)

        # 多尺度平均池化及注意力计算
        att1 = self.conv_atten1(self.avg_pool1(output))
        att2 = self.conv_atten2(self.avg_pool2(output))
        att3 = self.conv_atten3(self.avg_pool3(output))

        # 上采样使尺寸一致
        att2 = nn.functional.interpolate(att2, size=att1.size()[2:], mode='bilinear', align_corners=True)
        att3 = nn.functional.interpolate(att3, size=att1.size()[2:], mode='bilinear', align_corners=True)

        # 融合多尺度注意力
        att = att1 + att2 + att3

        # 门控机制
        gate = self.gate_conv(output)
        att = att * gate

        # 应用注意力权重
        output = output * att

        # 减少通道数量
        output = self.conv_redu(output)

        # 残差分支
        residual = self.residual_conv(x)
        output = output + residual

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
    # 初始化创新 DFF 模块
    model = InnovativeDFF(32)
    # 前向传播
    output = model(input1, input2)
    print("InnovativeDFF_input size:", input1.size())
    print("InnovativeDFF_Output size:", output.size())