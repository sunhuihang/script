import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 微信公众号 / B站: CV缝合救星

class CLC(nn.Module):
    """
    CLCk: Conv k×k -> LeakyReLU -> Conv k×k
    """
    def __init__(self, in_ch, out_ch=None, k=3, negative_slope=0.1):                                                                                                                                                                     # 微信公众号:CV缝合救星
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),                                                                                                                                                                     # 微信公众号:CV缝合救星
        )

    def forward(self, x):
        return self.net(x)

def _choose_gn_groups(C: int) -> int:
    # 选择能被C整除的最大不超过32的组数（至少1）
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1

class DNRU(nn.Module):
    """
    DNRU: 深度卷积 3×3 + GroupNorm + ReLU + 上采样
    """
    # 设置up_scale=1, 不进行上采样，保障输入和输出大小一致, 如需要输出特征图是输入的2倍, 可设置 up_scale=2或其它                                                                                                                                                                     # 微信公众号:CV缝合救星
    def __init__(self, channels, up_scale=1):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        self.gn = nn.GroupNorm(_choose_gn_groups(channels), channels)                                                                                                                                                                     # 微信公众号:CV缝合救星
        self.relu = nn.ReLU(inplace=True)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.relu(x)
        if self.up_scale and self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        return x

# 通道向量 <-> 2D 网格 的 reshape 工具（为了做 2D FFT，将 C 维通道描述铺成接近方形的网格）
def vector_to_grid(x_vec):
    """
    x_vec: (B, C, 1, 1) -> (B, 1, Hc, Wc), 同时返回 (Hc, Wc, C, pad)                                                                                                                                                                     # 微信公众号:CV缝合救星
    """
    B, C, _, _ = x_vec.shape
    Hc = int(math.floor(math.sqrt(C)))
    Wc = int(math.ceil(C / Hc))
    pad = Hc * Wc - C
    if pad > 0:
        x_vec = F.pad(x_vec.view(B, C), (0, pad))  # 在通道描述末尾补零                                                                                                                                                                     # 微信公众号:CV缝合救星
        C_ = C + pad
    else:
        x_vec = x_vec.view(B, C)
        C_ = C
    grid = x_vec.view(B, 1, Hc, Wc)
    return grid, (Hc, Wc, C, pad)


def grid_to_vector(grid, meta):
    """
    grid: (B, 1, Hc, Wc) -> (B, C, 1, 1)
    """
    Hc, Wc, C, pad = meta
    B = grid.size(0)
    vec = grid.view(B, Hc * Wc)
    if pad > 0:
        vec = vec[:, :C]
    return vec.view(B, C, 1, 1)


# 傅里叶残差通道注意力 (FRCA)
class FourierResidualChannelAttention(nn.Module):
    def __init__(self, channels, negative_slope=0.1, up_scale=1):                                                                                                                                                                     # 微信公众号:CV缝合救星
        super().__init__()
        self.channels = channels

        # 前端特征提取（对应图里的 CLC3）
        self.clc3 = CLC(channels, channels, k=3, negative_slope=negative_slope)                                                                                                                                                                     # 微信公众号:CV缝合救星

        # 对振幅/相位的轻量映射（对应 CLC1），这里作用在单通道 2D 网格上
        self.clc1_amp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )
        self.clc1_pha = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),                                                                                                                                                                     # 微信公众号:CV缝合救星
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )

        # DNRU
        self.dnru = DNRU(channels, up_scale=up_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, "channels mismatch"

        # CLC3 特征
        feat = self.clc3(x)  # (B, C, H, W)

        # GAP -> 通道描述向量
        chan_desc = F.adaptive_avg_pool2d(feat, 1)  # (B, C, 1, 1)

        # 铺成 2D 网格并做 2D FFT
        grid, meta = vector_to_grid(chan_desc)      # (B, 1, Hc, Wc)
        spec = torch.fft.fft2(grid)                 # 复数张量 (B, 1, Hc, Wc)                                                                                                                                                                     # 微信公众号:CV缝合救星
        amp = torch.abs(spec)
        pha = torch.angle(spec)

        # 对振幅/相位做 CLC1 调制
        amp = amp * self.clc1_amp(amp)
        pha = pha * self.clc1_pha(pha)

        # Complex：由 (amp, pha) 复合为新的频谱
        spec_new = torch.polar(amp, pha)

        # 2D ICFFT -> 得到调制后的网格，再还原为通道向量
        grid_ifft = torch.fft.ifft2(spec_new).real   # (B, 1, Hc, Wc)
        weight_vec = grid_to_vector(grid_ifft, meta) # (B, C, 1, 1)
        weight = torch.sigmoid(weight_vec)           # channel-wise 权重

        # 注意力 + 残差
        y = feat * weight
        out = y + x 

        # DNRU
        out = self.dnru(out)
        return out

if __name__ == "__main__":
    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W)

    module = FourierResidualChannelAttention(channels=C, up_scale=1) # 由于 DNRU 上采样 ×2，此处暂时不设置上采样即up_scale=1, 保证输入输出大小一致
    y = module(x)

    print(module)
    print("\n哔哩哔哩/微信公众号:CV缝合救星\n")
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}") 