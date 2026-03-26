import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt


class MSC_Modified(nn.Module):
    def __init__(self, dim, num_heads=8, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(MSC_Modified, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        # 可学习 TopK 比率参数
        self.k_ratio1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.k_ratio2 = nn.Parameter(torch.tensor(0.25), requires_grad=True)

        # 可变卷积代替池化
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel[0], stride=s[0], padding=pad[0], groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel[1], stride=s[1], padding=pad[1], groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=kernel[2], stride=s[2], padding=pad[2], groups=dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        B, C, H, W = y.shape

        # 多尺度卷积 + 双线性插值融合
        y1 = self.conv1(y)
        y2 = self.conv2(y)
        y3 = self.conv3(y)

        # 将不同尺度特征上采样到同一大小后加和
        y2 = nn.functional.interpolate(y2, size=y1.shape[2:], mode='bilinear', align_corners=False)
        y3 = nn.functional.interpolate(y3, size=y1.shape[2:], mode='bilinear', align_corners=False)
        y = y1 + y2 + y3

        y = rearrange(y, 'b c h w -> b (h w) c')
        y = self.layer_norm(y)

        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N, C = x.shape
        N1 = y.shape[1]

        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 动态 TopK
        k1_num = (N1 * self.sigmoid(self.k_ratio1)).int().clamp(1, N1)
        k2_num = (N1 * self.sigmoid(self.k_ratio2)).int().clamp(1, N1)

        def topk_mask(attn, k_num):
            mask = torch.zeros_like(attn)
            index = torch.topk(attn, k=k_num.item(), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn_masked = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
            attn_masked = attn_masked.softmax(dim=-1)
            attn_masked = self.attn_drop(attn_masked)
            return attn_masked

        attn1 = topk_mask(attn, k1_num)
        attn2 = topk_mask(attn, k2_num)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)

        # 加权融合
        out = 0.6 * out1 + 0.4 * out2

        x_out = out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        # 残差连接
        x_out = x_out + x

        hw = int(sqrt(N))
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=hw, w=hw)

        return x_out


if __name__ == "__main__":
    # 测试代码
    batch_size = 1
    channels = 32
    height = 64
    width = 64

    x = torch.randn(batch_size, channels, height, width).cuda()
    y = torch.randn(batch_size, channels, height, width).cuda()

    model = MSC_Modified(dim=channels, num_heads=8).cuda()
    output = model(x, y)

    print(model)
    print("🔥 哔哩哔哩：CV缝合救星！")
    print("输入 x 形状:", x.shape)
    print("输入 y 形状:", y.shape)
    print("输出形状   :", output.shape)
