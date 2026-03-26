import torch
import torch.nn as nn
import torch.nn.functional as F

# 哔哩哔哩：CV缝合救星
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class KernelSelectiveFusionAttention(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        # 多分支卷积
        self.conv_branch1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_branch2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_branch3 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            Mish()  # 使用 Mish 激活函数
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        # 位置注意力相关
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, 1))
        self.pos_conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)# 哔哩哔哩：CV缝合救星
        height, width = x.size(2), x.size(3)

        # 添加位置嵌入
        pos_embedded_x = x + self.pos_embedding.expand(-1, -1, height, width)
        pos_attn = self.pos_conv(pos_embedded_x)
        pos_attn = torch.sigmoid(pos_attn)

        # 多分支卷积结果相加
        attn_branch1 = self.conv_branch1(x)
        attn_branch2 = self.conv_branch2(x)
        attn_branch3 = self.conv_branch3(x)
        attn1 = attn_branch1 + attn_branch2 + attn_branch3

        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)# 哔哩哔哩：CV缝合救星

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        ch_attn1 = self.global_pool(attn)
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)

        # 动态调整
        std = torch.std(ch_attn1, dim=1, keepdim=True)
        a_b = a_b * (1 + std)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)

        a1, a2 = a_b.chunk(2, dim=1)# 哔哩哔哩：CV缝合救星
        a1 = a1.reshape(batch_size, dim // 2, 1, 1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()

        # 融合位置注意力
        attn = attn * pos_attn

        # 引入自适应参数 alpha
        alpha = 0.5  # 这里可以根据实际情况动态调整
        output = alpha * x + (1 - alpha) * attn

        return output


if __name__ == '__main__':
    batch_size = 2
    dim = 32
    height, width = 256, 256
    x = torch.randn(batch_size, dim, height, width)
    model = KernelSelectiveFusionAttention(dim=dim)
    print(model)
    output = model(x)
    # 哔哩哔哩：CV缝合救星
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")
