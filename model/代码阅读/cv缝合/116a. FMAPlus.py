import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------
# Utils
# -----------------------------
def window_partition(x, window_size):
    """
    x: [B, C, H, W]
    return: windows [B*nH*nW, C, Wh, Ww], (Hp, Wp, pad)
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")  # pad (left,right,top,bottom)

    Hp, Wp = x.shape[-2:]
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return x, Hp, Wp, (pad_h, pad_w)

def window_unpartition(windows, Hp, Wp, window_size, pad):
    """
    windows: [B*nH*nW, C, Wh, Ww]
    return: x: [B, C, H, W]
    """
    pad_h, pad_w = pad
    B_ = windows.shape[0] // ((Hp // window_size) * (Wp // window_size))
    C = windows.shape[1]
    x = windows.view(B_, Hp // window_size, Wp // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B_, C, Hp, Wp)
    if pad_h or pad_w:
        x = x[:, :, :Hp - pad_h, :Wp - pad_w]
    return x

# -----------------------------
# Norm (channels_first)
# -----------------------------
class LayerNorm(nn.Module):
    """From ConvNeXt (channels_first)"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ["channels_last", "channels_first"]
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# -----------------------------
# Enhanced Fourier Unit with Spectral Gating
# -----------------------------
class FourierUnitEnhanced(nn.Module):
    """
    - 频域 1x1 卷积（实+虚拼接）
    - 幅度谱门控（降低无关频率）
    - 轻量瓶颈 (bottleneck_ratio) 降低参数
    """
    def __init__(self, dim, groups=1, fft_norm='ortho', bottleneck_ratio=0.5):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm
        mid = max(1, int(dim * bottleneck_ratio))

        self.proj_in = nn.Conv2d(dim * 2, mid * 2, 1, bias=False, groups=self.groups)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(mid * 2, dim * 2, 1, bias=False, groups=self.groups)

        # 幅度谱门控：对 |F(x)| 进行 1x1 -> GELU -> 1x1 -> Sigmoid
        self.mag_gate = nn.Sequential(
            nn.Conv2d(dim, max(1, dim // 2), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(1, dim // 2), dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # 频域变换（rfft2 输出复数）
        Xf = torch.fft.rfft2(x, norm=self.fft_norm)  # [B, C, H, W//2 + 1], complex

        # 频谱门控：用幅度谱生成 gate，然后对实部/虚部共同缩放
        mag = torch.abs(Xf)  # [B, C, H, W//2+1]
        # 将 W//2+1 当成“宽度”处理，直接用 conv2d：需要拼成实数通道
        mag_gate = self.mag_gate(mag)  # [B, C, H, W//2+1]
        Xf = Xf * mag_gate  # broadcast on complex

        # 实部/虚部分离并拼接
        real = Xf.real
        imag = Xf.imag
        cat = torch.cat([real, imag], dim=1)  # [B, 2C, H, W//2+1]

        y = self.proj_in(cat)
        y = self.act(y)
        y = self.proj_out(y)

        # 还原为复数
        real2, imag2 = torch.chunk(y, 2, dim=1)
        Xf2 = torch.complex(real2, imag2)

        # 逆变换
        out = torch.fft.irfft2(Xf2, s=(H, W), norm=self.fft_norm)
        return out

# -----------------------------
# FMA++ : Windowed Fourier Modulated Attention
# -----------------------------
class FMAPlus(nn.Module):
    """
    创新点：
      - 按窗口做 FMA（区域化频率-空间调制）
      - 频域门控 + 轻量瓶颈
      - Head-wise Temperature 可学习
      - Value 分支：1x1 + DWConv 提升局部细节
    形状：
      输入/输出: [B, C, H, W]
    """
    def __init__(self, dim, num_heads=4, window_size=8, bottleneck_ratio=0.5):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # Fourier branch (acts as 'A')
        self.fourier = FourierUnitEnhanced(dim, bottleneck_ratio=bottleneck_ratio)

        # Value branch: 1x1 -> DWConv -> GELU -> 1x1（轻量）
        self.v_proj1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_act = nn.GELU()
        self.v_proj2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Head-wise temperature（>0）：用 softplus 确保正数
        self._temp = nn.Parameter(torch.ones(num_heads))  # 初值 1
        self.softplus = nn.Softplus()

        # CPE & output projection
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # 可选层缩放
        self.layer_scale = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x
        pos = self.cpe(x)
        x = self.norm(x)

        # A：Fourier global-ish features（仍在空间域）
        A = self.fourier(x)  # [B, C, H, W]

        # V：局部增强 value
        V = self.v_proj1(x)
        V = self.v_dw(V)
        V = self.v_act(V)
        V = self.v_proj2(V)

        # 分窗口
        w = self.window_size
        A_win, Hp, Wp, pad = window_partition(A, w)  # [B*nWin, C, w, w]
        V_win, _, _, _ = window_partition(V, w)

        # head 分组 -> [B*nWin, head, ch, w*w]
        A_tok = rearrange(A_win, 'bn (head ch) h w -> bn head ch (h w)', head=self.num_heads)
        V_tok = rearrange(V_win, 'bn (head ch) h w -> bn head ch (h w)', head=self.num_heads)

        # 元素乘：线性注意力（无需 QK^T）
        # 引入 head-wise temperature 调整锐度
        temp = self.softplus(self._temp).view(1, self.num_heads, 1, 1)
        attn = (A_tok * V_tok) * temp  # [bn, head, ch, tokens]

        # 归一化（每个 head、每个通道在 token 维度做 softmax）
        attn = F.softmax(attn, dim=-1)

        # 还原为窗口图像 [bn, C, w, w]
        Xw = rearrange(attn, 'bn head ch (h w) -> bn (head ch) h w', head=self.num_heads, h=w, w=w)

        # 反窗口拼回 & 去掉 padding
        X = window_unpartition(Xw, Hp, Wp, w, pad)  # [B, C, H, W]

        # 残差 + 投影
        X = X + pos
        X = self.proj(X)

        # layer_scale（逐通道缩放）
        X = self.layer_scale.view(1, -1, 1, 1) * X

        out = X + shortcut
        return out

# -----------------------------
# 简单测试
# -----------------------------
if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 1, 32, 256, 256
    heads = 4
    win = 8  # 不必整除，模块会自动 padding

    x = torch.randn(B, C, H, W).to(device)
    model = FMAPlus(dim=C, num_heads=heads, window_size=win, bottleneck_ratio=0.5).to(device)

    with torch.no_grad():
        y = model(x)

    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    # 简单一致性检查
    assert y.shape == x.shape, "输出形状应与输入一致！"
