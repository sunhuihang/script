import torch
import torch.nn as nn
from transformers import ViTConfig
from timm.models.layers import DropPath

#哔哩哔哩 CV缝合救星
class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):#哔哩哔哩 CV缝合救星
        out = x + self.bias.expand_as(x)
        return out


class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out#哔哩哔哩 CV缝合救星


class MSGDC(nn.Module):
    def __init__(self, in_chn, config, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(MSGDC, self).__init__()
        self.move = LearnableBiasnn(in_chn)
        self.cov1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation=dilation1, bias=True)
        self.cov2 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation=dilation2, bias=True)
        self.cov3 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation=dilation3, bias=True)
        self.norm = nn.LayerNorm(in_chn)
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)#哔哩哔哩 CV缝合救星

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.move(x)
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
    config = ViTConfig()
    # 创建 DeformableInteractiveAttention 模型实例
    model = MSGDC(C,config).cuda()

    # 打印模型结构
    print(model)
    print("哔哩哔哩: CV缝合救星!")

    # 前向传播
    output = model(x)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")  # 打印输入张量的形状
    print(f"输出张量的形状: {output.shape}")  # 打印输出张量的形状
