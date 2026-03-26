import torch
import torch.nn as nn
import torch.nn.functional as F

class DualCoordAttentionGate(nn.Module):
    def __init__(self, inp, oup, groups=32, reduction=4, use_residual=True):
        super(DualCoordAttentionGate, self).__init__()
        self.use_residual = use_residual

        # 水平与垂直方向的平均/最大池化
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        # 压缩中间维度 mip
        mip = max(8, inp // groups)

        # 平均和最大通路共享的卷积模块（降低通道维度）
        self.shared_conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.shared_bn = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)

        # 分别用于水平和垂直方向的卷积恢复通道维度
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

        # 可学习门控机制，用于自适应融合 mean 与 max 注意力
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // reduction, 2, 1),  # 输出两个权重：mean 和 max
            nn.Softmax(dim=1)  # 对两个权重进行归一化
        )

        # 通道注意力模块（SE 类似）增强融合后的特征响应
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, oup // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup // reduction, oup, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 平均池化路径
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.shared_conv1(y_mean)
        y_mean = self.shared_bn(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)
        x_h_mean = self.conv_h(x_h_mean).sigmoid()
        x_w_mean = self.conv_w(x_w_mean).sigmoid()
        attn_mean = x_h_mean * x_w_mean

        # 最大池化路径
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.shared_conv1(y_max)
        y_max = self.shared_bn(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)
        x_h_max = self.conv_h(x_h_max).sigmoid()
        x_w_max = self.conv_w(x_w_max).sigmoid()
        attn_max = x_h_max * x_w_max

        # 可学习权重融合 mean 和 max 路径
        gate_weights = self.gate(identity)  # 输出维度为 [B, 2, 1, 1]
        mean_weight = gate_weights[:, 0:1]
        max_weight = gate_weights[:, 1:2]
        attn = attn_mean * mean_weight + attn_max * max_weight

        # 应用注意力
        out = identity * attn

        # 通道注意力增强输出特征
        scale = self.channel_att(out)
        out = out * scale

        # 可选残差连接
        if self.use_residual:
            out = out + identity

        return out

if __name__ == "__main__":

    # 输入参数设置
    batch_size = 1
    channels = 32
    height = 256
    width = 256

    # 构造输入张量
    x = torch.randn(batch_size, channels, height, width).cuda()

    # 实例化模块
    model = DualCoordAttentionGate(inp=channels, oup=channels).cuda()
    print(model)
    print("哔哩哔哩：CV缝合救星！")

    # 前向传播
    out = model(x)

    # 打印输入输出张量的形状
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
