import torch
import torch.nn as nn


class AdaptiveMultiScaleShuffleAttention(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, act_layer=nn.GELU, input_resolution=(64, 64)):
        super().__init__()
        self.input_resolution = input_resolution
        self.in_features = in_features
        self.out_features = out_features

        # 多尺度卷积层
        self.conv1x1 = nn.Conv2d(in_features, in_features // 2, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_features, in_features // 2, kernel_size=3, stride=1, padding=1)

        # 定义 gating 部分，使用平均池化后经过卷积层和 Sigmoid 激活
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化，输出大小为 1x1
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0),  # 卷积层
            nn.Sigmoid()  # Sigmoid 激活函数
        )

    # 通道重排操作，打乱输入张量的通道
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()  # 获取输入张量的维度
        group = self.dynamic_grouping(x)  # 动态分组
        assert num_channels % group == 0  # 确保通道数可以被分组数整除
        group_channels = num_channels // group  # 每个组的通道数

        # 将输入张量 reshape 成 (batch_size, group_channels, group, height, width)
        x = x.reshape(batchsize, group_channels, group, height, width)
        # 调整维度顺序，使得每个组的通道打乱
        x = x.permute(0, 2, 1, 3, 4)
        # 将张量恢复成 (batch_size, num_channels, height, width)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    # 动态分组机制
    def dynamic_grouping(self, x):
        # 简单示例：根据输入特征的均值动态调整分组数
        mean_value = x.mean().item()
        if mean_value > 0.5:
            return 4
        else:
            return 2

    # 前向传播函数
    def forward(self, x):
        y = x  # 保存输入张量，用于残差连接

        # 多尺度特征融合
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x = torch.cat([x1, x2], dim=1)

        x = self.channel_shuffle(x)  # 对输入进行通道重排
        x = self.gating(x)  # 使用 gating 对输入进行处理

        return y * x  # 将原始输入与处理后的输出相乘


from torchinfo import summary  # 需要安装 torchinfo：pip install torchinfo

if __name__ == '__main__':
    # 设置输入参数
    batch_size = 1  # 批次大小
    in_channels = 32  # 输入通道数
    out_channels = 32  # 输出通道数
    input_resolution = (256, 256)  # 输入分辨率

    # 创建随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1]).cuda()  # 输入张量

    # 创建 AdaptiveMultiScaleShuffleAttention 模块
    model = AdaptiveMultiScaleShuffleAttention(in_features=in_channels, out_features=out_channels,
                                               input_resolution=input_resolution).cuda()

    # 使用 torchinfo 进行模型分析
    summary(model, input_size=(batch_size, in_channels, input_resolution[0], input_resolution[1]))
    print("\n哔哩哔哩: CV缝合救星!\n")

    # 前向传播
    output = model(x)

    # 打印输入和输出张量的形状
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
