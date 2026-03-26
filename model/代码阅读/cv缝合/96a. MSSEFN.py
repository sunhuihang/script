import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# ---------------- LayerNorm ----------------
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# ---------------- MS-SEFN ----------------
class MSSEFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=True):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 多尺度空间增强模块：主通路 + 小尺度通路
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            LayerNorm(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            LayerNorm(dim), nn.ReLU(inplace=True)
        )
        self.conv_small = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )

    def forward(self, x, spatial):
        x = self.project_in(x)

        y_main = self.avg_pool(spatial)
        y_main = self.conv_main(y_main)
        y_main = F.interpolate(y_main, size=x.shape[2:], mode='bilinear', align_corners=False)

        y_small = self.conv_small(spatial)

        y = y_main + y_small

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# ---------------- 测试入口 ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W).to(device)
    spatial = torch.randn(B, C, H, W).to(device)

    model = MSSEFN(dim=C, ffn_expansion_factor=4, bias=True).to(device)
    out = model(x, spatial)

    print(f"Input shape: {x.shape}")
    print("\n哔哩哔哩：CV缝合救星!\n")
    print(f"Spatial shape: {spatial.shape}")
    print(f"Output shape: {out.shape}")
