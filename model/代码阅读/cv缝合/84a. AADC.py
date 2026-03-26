import torch
import torch.nn as nn
from transformers import ViTConfig
from timm.models.layers import DropPath

#哔哩哔哩 CV缝合救星
class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class RPReLU(nn.Module):
    def __init__(self, hidden_size):#哔哩哔哩 CV缝合救星
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out#哔哩哔哩 CV缝合救星
        return self.sigmoid(out)


class AADC(nn.Module):
    def __init__(self, in_chn, config, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(AADC, self).__init__()
        self.move = LearnableBiasnn(in_chn)
        self.config = config
        self.attn = ChannelAttention(in_chn)#哔哩哔哩 CV缝合救星

        self.dilation1 = nn.Parameter(torch.tensor(dilation1, dtype=torch.float32), requires_grad=True)
        self.dilation2 = nn.Parameter(torch.tensor(dilation2, dtype=torch.float32), requires_grad=True)
        self.dilation3 = nn.Parameter(torch.tensor(dilation3, dtype=torch.float32), requires_grad=True)

        self.cov1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation=1, bias=True)
        self.cov2 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation=1, bias=True)
        self.cov3 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation=1, bias=True)

        self.norm = nn.LayerNorm(in_chn)
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.move(x)

        # 自适应调整扩张率
        dilation1 = torch.clamp(self.dilation1, min=1).int().item()
        dilation2 = torch.clamp(self.dilation2, min=1).int().item()
        dilation3 = torch.clamp(self.dilation3, min=1).int().item()

        self.cov1.dilation = (dilation1, dilation1)
        self.cov2.dilation = (dilation2, dilation2)
        self.cov3.dilation = (dilation3, dilation3)

        x = self.attn(x) * x

        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1, 2)
        x1 = self.act1(x1)
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1, 2)
        x2 = self.act2(x2)#哔哩哔哩 CV缝合救星
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1, 2)
        x3 = self.act3(x3)
        x = self.norm(x1 + x2 + x3)
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()


if __name__ == '__main__':
    # 设置输入张量的尺寸
    B, C, H, W = 1, 32, 256, 256  # 批量大小 B, 输入通道数 C, 高度 H, 宽度 W
    x = torch.randn(B, C, H, W).cuda()  # 创建输入张量，形状为 (B, C, H, W)，并将其移到 GPU
    config = ViTConfig()#哔哩哔哩 CV缝合救星
    # 创建 AADC 模型实例
    model = AADC(C, config).cuda()

    # 打印模型结构
    print(model)
    print("哔哩哔哩: CV缝合救星!")

    # 前向传播
    output = model(x)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")  # 打印输入张量的形状
    print(f"输出张量的形状: {output.shape}")  # 打印输出张量的形状