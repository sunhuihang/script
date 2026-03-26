import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------
# 工具层：深度可分离卷积
# ---------------------------
class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.SiLU()
        )


# ---------------------------
# 工具层：残差块
# ---------------------------
class IdentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, rate=1):
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            SeparableConvBNReLU(F1, F2, kernel_size, dilation=rate),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


# ---------------------------
# 相对位置编码（轻量级实现）
# ---------------------------
class RelPosEnc(nn.Module):
    def __init__(self, channels):
        super(RelPosEnc, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)

    def forward(self, x):
        return self.dwconv(x)


# ---------------------------
# Pairwise 跨尺度注意力
# ---------------------------
class CrossScaleAttention(nn.Module):
    def __init__(self, dim=128):
        super(CrossScaleAttention, self).__init__()
        self.qkv = nn.Conv2d(dim * 2, dim * 3, 1, bias=False)
        self.scale = dim ** -0.5
        self.proj = nn.Conv2d(dim, dim, 1)
        self.rpe = RelPosEnc(dim)

    def forward(self, x, y):
        b, c, h, w = x.shape
        feat = torch.cat([x, y], dim=1)  # 拼接后生成 qkv
        qkv = self.qkv(feat)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b (h w) c")
        v = rearrange(v, "b c h w -> b (h w) c")

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        out = self.proj(out + self.rpe(y))
        return out


# ---------------------------
# CSAGBlock 主体
# ---------------------------
class CSAGBlock(nn.Module):
    """
    Cross-Scale Attention Gated Block
    """
    def __init__(self, dim_out=128):
        super(CSAGBlock, self).__init__()
        # 四个分支
        self.branch1 = nn.Sequential(  # 输入 x1: 32通道
            SeparableConvBNReLU(32, dim_out, 3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, 3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, 3, stride=1)
        )
        self.branch2 = nn.Sequential(  # 输入 x2: 64通道
            SeparableConvBNReLU(64, dim_out, 3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, 3, stride=1)
        )
        self.branch3 = SeparableConvBNReLU(128, dim_out, 3, stride=1)  # 输入 x3: 128通道
        self.branch4 = nn.Sequential(  # 输入 x4: 256通道
            nn.Conv2d(256, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(True)
        )

        # 融合卷积
        self.merge = nn.Sequential(
            nn.Conv2d(4 * dim_out, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(True)
        )
        self.resblock = nn.Sequential(
            IdentityBlock(dim_out, 3, [dim_out, dim_out, dim_out]),
            IdentityBlock(dim_out, 3, [dim_out, dim_out, dim_out])
        )

        # 跨尺度注意力模块
        self.attn = CrossScaleAttention(dim_out)

        # 门控
        self.gate = nn.Sequential(
            nn.Conv2d(dim_out, 1, 1),
            nn.Sigmoid()
        )

        # 输出卷积
        self.conv_out = nn.Conv2d(dim_out, dim_out, 1)

    def forward(self, x1, x2, x3, x4):
        # 四分支
        f1 = self.branch1(x1)  # 32 -> 128
        f2 = self.branch2(x2)  # 64 -> 128
        f3 = self.branch3(x3)  # 128 -> 128
        f4 = self.branch4(x4)  # 256 -> 128

        merge = self.merge(torch.cat([f1, f2, f3, f4], dim=1))
        merge = self.resblock(merge)

        # 两两交互
        att12 = self.attn(f1, f2)
        att34 = self.attn(f3, f4)

        # 门控融合
        g1 = self.gate(att12)
        g2 = self.gate(att34)

        out = merge + g1 * att12 + g2 * att34
        return self.conv_out(out)


# ---------------------------
# 测试代码
# ---------------------------
if __name__ == "__main__":
    dim_out = 128
    model = CSAGBlock(dim_out=dim_out)

    x1 = torch.randn(1, 32, 256, 256)   # 输入1
    x2 = torch.randn(1, 64, 128, 128)   # 输入2
    x3 = torch.randn(1, 128, 64, 64)    # 输入3
    x4 = torch.randn(1, 256, 64, 64)    # 输入4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)

    out = model(x1, x2, x3, x4)
    # 输出模型结构与形状信息
    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量 x_in1 形状:", x1.shape) 
    print("输入张量 x_in2 形状:", x2.shape) 
    print("输入张量 x_in3 形状:", x3.shape) 
    print("输入张量 x_in4 形状:", x4.shape)  
    print("输出张量形状       :", out.shape) 
