import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        weight = self.fc(self.avg_pool(x))
        return x * weight

class CompactSelfAttentionV2(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, sample_rate=2):
        super().__init__()
        self.num_heads = num_heads
        self.sample_rate = sample_rate
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.sampler = nn.AvgPool2d(1, stride=sample_rate)
        self.qkv = nn.Conv2d(dim // 2, dim // 2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, kernel_size=3, padding=1, groups=dim // 2 * 3, bias=bias)

        self.LocalProp = nn.ConvTranspose2d(dim // 2, dim, kernel_size=sample_rate, stride=sample_rate,
                                            padding=(sample_rate // 2), groups=1, bias=bias)

        self.channel_attn = SEBlock(dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.sample_rate - H % self.sample_rate) % self.sample_rate
        pad_w = (self.sample_rate - W % self.sample_rate) % self.sample_rate
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_sampled = self.sampler(x_padded)
        x1, x2 = x_sampled.chunk(2, dim=1)

        def self_attention(x_feat):
            qkv = self.qkv_dwconv(self.qkv(x_feat))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
            k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
            v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
            return rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads,
                             h1=x_feat.shape[2], w1=x_feat.shape[3])

        out1 = self_attention(x1)
        out2 = self_attention(x2)

        combined = torch.cat([out1, out2], dim=1)  # [B, dim, h, w]
        gate_weight = self.gate(combined)          # [B, dim//2, h, w]

        fused = gate_weight * out1 + (1 - gate_weight) * out2  # [B, dim//2, h, w]

        enhanced = self.LocalProp(fused)                      # ↑通道恢复为 dim
        enhanced = self.channel_attn(enhanced)
        enhanced = self.project_out(enhanced)
        enhanced = self.norm(enhanced)
        enhanced = enhanced[:, :, :H, :W]  # 裁剪填充部分

        return enhanced

# ✅ 测试入口
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 32, 256, 256).to(device)
    model = CompactSelfAttentionV2(dim=32, num_heads=8, sample_rate=2).to(device)
    out = model(x)
    print(model)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
