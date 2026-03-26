import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FreqAttnPlus(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(FreqAttnPlus, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.mid_dim = in_dim // 2

        self.reduce = nn.Conv2d(in_dim, self.mid_dim, 1)
        self.depthwise = nn.Conv2d(self.mid_dim, self.mid_dim, 3, padding=1, groups=self.mid_dim)
        self.norm = nn.BatchNorm2d(self.mid_dim)
        self.relu = nn.ReLU(inplace=True)

        # learnable gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.mid_dim, self.mid_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(self.mid_dim // 8, self.mid_dim, 1),
            nn.Sigmoid()
        )

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.post_proj = nn.Conv2d(self.mid_dim * 2, in_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_r = self.relu(self.norm(self.depthwise(self.reduce(x))))  # B, C/2, H, W

        fft_feat = torch.fft.fft2(x_r.float(), norm='ortho')  # complex
        q = k = v = fft_feat

        # reshape into heads
        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = F.softmax(attn.real, dim=-1)  # 只用实部参与 softmax

        out = attn @ v.real
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads, h1=H, w1=W)
        out = torch.fft.ifft2(out, norm='ortho').real  # 频域回到空间

        # 残差增强路径（门控频率残差）
        fwm = torch.fft.ifft2(fft_feat * torch.fft.fft2(self.gate(x_r) * x_r)).real
        fwm = torch.cat([out, fwm], dim=1)

        out = self.post_proj(fwm) + x  # 加残差
        return out

# 用于测试模块功能
if __name__ == "__main__":
    x = torch.randn(1, 32, 256, 256).cuda()
    model = FreqAttnPlus(32).cuda()
    print(model)
    with torch.no_grad():
        y = model(x)
    print("输入维度：", x.shape)
    print("\n哔哩哔哩：CV缝合救星！FreAttn++\n")
    print("输出维度：", y.shape)
