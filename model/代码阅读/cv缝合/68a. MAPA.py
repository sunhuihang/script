import torch
import torch.nn as nn

# B站:CV缝合救星
"""
CV缝合救星魔改创新：MAPA MultiScaleAdaptivePolarAttention（多尺度自适应极性注意力）
一、魔改创新点：
1. 引入多尺度特征融合：在计算注意力之前，通过不同卷积核大小的卷积层提取多尺度特征，
然后将这些特征融合，丰富特征表达。
2. 自适应注意力权重调整：引入一个自适应的权重调整机制，根据输入特征的统计信息动态
调整注意力权重。
3. 残差连接增强：在输出阶段增加额外的残差连接，帮助模型更好地学习特征。
二、具体实现
1.MultiScaleFeatureExtractor 类：通过两个不同卷积核大小的卷积层提取多尺度特征，然后
将这些特征拼接并通过一个卷积层进行融合。
2. AdaptiveAttentionWeightAdjustment 类：使用自适应平均池化和全连接层计算注意力权重
调整因子，然后将该因子应用于查询、键和值向量。
3. EnhancedPolaLinearAttention 类：在原有的PolaLinearAttention基础上，增加了多尺度
特征提取和自适应注意力权重调整的功能，并在输出阶段增加了残差连接。
"""

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.fusion(x_fused)
        return x_fused

class AdaptiveAttentionWeightAdjustment(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 16, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        y = self.avg_pool(q).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        q = q * y
        k = k * y
        v = v * y
        return q, k, v

class EnhancedPolaLinearAttention(nn.Module):
    def __init__(self, dim, H, W, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.feature_extractor = MultiScaleFeatureExtractor(dim)
        self.adaptive_weight = AdaptiveAttentionWeightAdjustment(dim)

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, dim, H // sr_ratio, W // sr_ratio)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, alpha, kernel_size))

    def forward(self, x):
        # 多尺度特征提取
        x = self.feature_extractor(x)
        B, C, H, W = x.shape
        N = H * W
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)

        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding.flatten(2).transpose(1, 2)
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * torch.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        # 自适应注意力权重调整
        q, k, v = self.adaptive_weight(q.permute(0, 2, 3, 1).reshape(B, C, H, W),
                                       k.permute(0, 2, 3, 1).reshape(B, C, H, W),
                                       v.permute(0, 2, 3, 1).reshape(B, C, H, W))
        q = q.permute(0, 3, 1, 2).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 3, 1, 2).reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 3, 1, 2).reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, self.head_dim, H, W)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        # 残差连接增强
        residual = x
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + residual

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


if __name__ == "__main__":
    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建测试输入张量 (batch_size, channels, height, width) / B C H W
    x = torch.randn(3, 128, 32, 32).to(device)
    # 初始化 pla 模块
    pla = EnhancedPolaLinearAttention(dim=128, H=32, W=32, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                                      kernel_size=5, alpha=4)
    print(pla)
    print("B站:CV缝合救星")
    pla = pla.to(device)
    # 前向传播
    output = pla(x)

    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)