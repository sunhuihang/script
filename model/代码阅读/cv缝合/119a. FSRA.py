import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

# =========================
# DCT/IDCT (no extra deps)
# =========================
def _dct_mat(N: int, device=None, dtype=None):
    """Orthonormal DCT-II matrix of size (N, N)."""
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # [1, N]
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # [N, 1]
    mat = torch.cos((torch.pi / N) * (n + 0.5) * k)               # [N, N]
    mat *= torch.sqrt(torch.tensor(2.0 / N, device=device, dtype=dtype))
    mat[0, :] *= 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return mat  # [N, N]

def dct2(x: torch.Tensor) -> torch.Tensor:
    """
    2D DCT-II with orthonormal normalization over last two dims (H, W).
    x: [B, C, H, W] -> returns same shape
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    CH = _dct_mat(H, device, dtype)   # [H, H]
    CW = _dct_mat(W, device, dtype)   # [W, W]
    y = x.reshape(B * C, H, W)
    y = torch.matmul(CH, y)                         # DCT along H
    y = torch.matmul(y.transpose(1, 2), CW.T).transpose(1, 2)  # DCT along W
    return y.reshape(B, C, H, W)

def idct2(X: torch.Tensor) -> torch.Tensor:
    """
    2D IDCT (inverse of orthonormal DCT-II).
    """
    B, C, H, W = X.shape
    device, dtype = X.device, X.dtype
    CH = _dct_mat(H, device, dtype)   # [H, H]
    CW = _dct_mat(W, device, dtype)   # [W, W]
    y = X.reshape(B * C, H, W)
    y = torch.matmul(y.transpose(1, 2), CW).transpose(1, 2)      # inverse along W
    y = torch.matmul(CH.T, y)                                    # inverse along H
    return y.reshape(B, C, H, W)

# =========================
# Soft High-Pass Mask
# =========================
class SoftHighPass(nn.Module):
    """
    Learnable soft high-pass mask in frequency domain.
    - alpha_h, alpha_w \in (0,1): controls LF cutoff along H/W
    - beta > 0: controls transition sharpness (larger = sharper)
    Mask(u,v) = 1 - sigma(beta*(alpha_h - u_norm)) * sigma(beta*(alpha_w - v_norm))
    This approximates a soft rectangle low-frequency suppression.
    """
    def __init__(self, init_alpha=(0.25, 0.25), init_beta=12.0):
        super().__init__()
        ah, aw = init_alpha
        self.alpha_h = nn.Parameter(torch.tensor(float(ah)).clamp(1e-4, 1-1e-4))
        self.alpha_w = nn.Parameter(torch.tensor(float(aw)).clamp(1e-4, 1-1e-4))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(float(init_beta))))

    def forward(self, H: int, W: int, device, dtype):
        # normalized coords in [0,1)
        u = torch.arange(H, device=device, dtype=dtype) / H  # [H]
        v = torch.arange(W, device=device, dtype=dtype) / W  # [W]
        Uh = u.unsqueeze(1)                                  # [H,1]
        Vw = v.unsqueeze(0)                                  # [1,W]

        beta = torch.exp(self.log_beta) + 1e-6
        # low-frequency soft area ~ product of two sigmoids
        low_h = torch.sigmoid(beta * (self.alpha_h - Uh))    # [H,1]
        low_w = torch.sigmoid(beta * (self.alpha_w - Vw))    # [1,W]
        low_rect = low_h * low_w                             # [H,W]
        mask = 1.0 - low_rect                                # high-pass emphasis in [0,1]
        return mask  # [H, W]

# =========================
# FSRA Block (CVPR-style)
# =========================
class FSRA(nn.Module):
    """
    FSRA: Frequency-Selective Residual Attention (CVPR-style name)
    - 可学习软高通 + 频域通道门控 + 空间注意融合 + 残差
    """
    def __init__(self,
                 in_channels: int,
                 groups: int = 32,
                 dw_kernel: int = 5,
                 dw_dilation: int = 1,
                 init_alpha=(0.25, 0.25),
                 init_beta=12.0):
        super().__init__()
        self.in_channels = in_channels
        self.groups = max(1, min(groups, in_channels))  # 确保能整除
        while in_channels % self.groups != 0 and self.groups > 1:
            self.groups //= 2
        self.soft_hp = SoftHighPass(init_alpha=init_alpha, init_beta=init_beta)

        # 频域通道门控：对频域系数做全局池化 -> 1x1(groups) -> Sigmoid
        self.spec_channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups, bias=False),
            nn.Sigmoid()
        )

        # 空间分支：DWConv 捕获局部细节（可用大核或空洞）
        padding = ((dw_kernel - 1) // 2) * dw_dilation
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel,
                      padding=padding, dilation=dw_dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU()
        )

        # 频域->时域后的轻量 refine
        self.freq_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups, bias=False),
            nn.GELU()
        )

        # 输出融合
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(self.groups, in_channels), num_channels=in_channels)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        identity = x

        # ---------- Frequency branch ----------
        Xf = dct2(x)                                    # [B, C, H, W]
        mask = self.soft_hp(H=Xf.size(-2), W=Xf.size(-1), device=Xf.device, dtype=Xf.dtype)
        mask = mask.view(1, 1, Xf.size(-2), Xf.size(-1)).expand_as(Xf)  # [B,C,H,W]
        Xf = Xf * mask                                  # 软高通

        # 通道门控在频域：先把频域能量汇聚（GAP+GMP 可二选一；这里用 GAP）
        spec_pool = F.adaptive_avg_pool2d(Xf, output_size=(1, 1))  # [B,C,1,1]
        spec_gate = self.spec_channel_gate(spec_pool)               # [B,C,1,1]
        Xf = Xf * spec_gate                                         # 频域通道注意

        x_freq = idct2(Xf)                                          # 还原到时域
        x_freq = self.freq_refine(x_freq)                           # 轻量 refine

        # ---------- Spatial branch ----------
        x_spa = self.spatial_branch(x)

        # ---------- Fuse + Residual ----------
        out = self.out(x_freq + x_spa)
        return identity + out

# =========================
# Backward compatible HFP (wrapper)
# =========================
class HFP_FSRA(nn.Module):
    """
    用 FSRA 替代原 HFP 的“频-空间双路径”，接口保持简洁。
    """
    def __init__(self,
                 in_channels: int,
                 init_alpha=(0.25, 0.25),
                 init_beta=12.0,
                 groups=32,
                 dw_kernel=5,
                 dw_dilation=1):
        super().__init__()
        self.block = FSRA(in_channels=in_channels,
                          groups=groups,
                          dw_kernel=dw_kernel,
                          dw_dilation=dw_dilation,
                          init_alpha=init_alpha,
                          init_beta=init_beta)

    def forward(self, x):
        return self.block(x)

# =========================
# Simple test / demo
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)

    # 输入配置
    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W)
    # FSRA: Frequency-Selective Residual Attention
    model = HFP_FSRA(
        in_channels=C,
        init_alpha=(0.25, 0.25),   # 软高通的初始截止比例（可学习）
        init_beta=12.0,            # 软边界锐度（可学习）
        groups=32,                 # 分组卷积组数
        dw_kernel=7,               # 空间分支大核（可改小到5以提速）
        dw_dilation=1              # 空洞率（>1时扩大感受野）
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(model)
    print("\nCV缝合救星: FSRA (Frequency-Selective Residual Attention)")
    print("输入:", x.shape, "输出:", y.shape)
