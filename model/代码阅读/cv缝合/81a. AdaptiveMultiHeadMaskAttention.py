import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class AdaptiveMultiHeadMaskAttention(nn.Module):
    def __init__(self, channels, size, num_heads=4):
        super(AdaptiveMultiHeadMaskAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        # 改为动态掩码生成网络
        self.mask_generator = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, size[0] * size[1])
        )

        self.norm = nn.LayerNorm([channels])

        self.pos_encoding = nn.Parameter(torch.randn(1, size[0] * size[1], channels))

        self.residual_block_in = ResidualBlock(channels)
        self.residual_block_out = ResidualBlock(channels)

    def forward(self, x):
        x = self.residual_block_in(x)

        batch_size, channels, height, width = x.size()
        if channels != self.channels:
            raise ValueError("Input channel size does not match initialized channel size.")

        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
        x = x + self.pos_encoding

        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 动态生成掩码
        feature_mean = x.mean(dim=1)
        mask = self.mask_generator(feature_mean)
        # 调整mask形状与scores兼容
        mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_heads, height * width, height * width)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        scores = scores + mask

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, channels)

        attention_output = attention_output + x
        attention_output = self.norm(attention_output)

        attention_output = attention_output.view(batch_size, channels, height, width)
        attention_output = self.residual_block_out(attention_output)

        return attention_output


if __name__ == "__main__":
    batch_size = 2
    channels = 64
    height = 16
    width = 16

    x = torch.randn(batch_size, channels, height, width).cuda()
    attention_module = AdaptiveMultiHeadMaskAttention(channels=channels, size=(height, width)).cuda()

    print("AdaptiveMultiHeadMaskAttention模型结构:")
    print(attention_module)

    output = attention_module(x)

    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")