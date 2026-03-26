import torch
import torch.nn as nn
import torch.nn.functional as F


class NovaOp(nn.Module):
    """Nova模块内部的动态多尺度卷积感知单元"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

        # 可学习加权融合
        self.weight3 = nn.Parameter(torch.ones(1))
        self.weight5 = nn.Parameter(torch.ones(1))
        self.weight7 = nn.Parameter(torch.ones(1))

        # 小型残差块
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
        )

        self.projector = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        out7 = self.conv7x7(x)

        # 动态融合
        out = (self.weight3 * out3 + self.weight5 * out5 + self.weight7 * out7) / (
                    self.weight3 + self.weight5 + self.weight7 + 1e-6)
        out = out + identity

        # 小型残差进一步强化
        out = out + self.residual_conv(out)

        # 压缩输出
        out = self.projector(out)
        return out


class Nova(nn.Module):
    """Nova模块本体"""

    def __init__(self, in_dim):
        super().__init__()
        hidden_dim = 64

        self.norm = nn.LayerNorm(in_dim)
        self.scale_input = nn.Parameter(torch.ones(1))  # 输入增益
        self.shift_input = nn.Parameter(torch.zeros(1))  # 输入偏移

        self.down_proj = nn.Linear(in_dim, hidden_dim)
        self.nova_op = NovaOp(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.up_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)  # (B, HW, C)

        identity = x

        # 输入归一化 + 增益偏移
        x = self.scale_input * self.norm(x) + self.shift_input

        # 降维
        x = self.down_proj(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, hidden_dim, H, W)

        # 多尺度动态卷积
        x = self.nova_op(x)

        # 还原形状
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        # 激活 + dropout
        x = self.act(x)
        x = self.dropout(x)

        # 升维
        x = self.up_proj(x)

        # 残差连接
        out = identity + x
        out = out.transpose(1, 2).reshape(B, -1, H, W)  # (B, C, H, W)

        return out


# 测试代码
if __name__ == "__main__":
    # 实例化Nova模块，输入通道是32
    model = Nova(32)
    input_tensor = torch.randn(1, 32, 32, 32)  # B, C, H, W
    output_tensor = model(input_tensor)
    print(model)
    print("\n 哔哩哔哩：CV缝合救星!\n")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
