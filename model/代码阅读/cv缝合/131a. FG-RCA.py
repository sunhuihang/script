import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 工具: 选择 GroupNorm 分组
# ==============================
def _choose_gn_groups(C: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1


# ==============================
# 基础块: CLC (Conv-LeakyReLU-Conv)
# ==============================
class CLC(nn.Module):
    """
    CLCk: Conv k×k -> LeakyReLU -> Conv k×k
    """
    def __init__(self, in_ch, out_ch=None, k=3, negative_slope=0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# 轻量上采样块: DNRU
# ==============================
class DNRU(nn.Module):
    """
    DNRU: 深度可分离卷积 3×3 + GroupNorm + ReLU + 可选上采样
    """
    def __init__(self, channels, up_scale=1):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(channels), channels)
        self.relu = nn.ReLU(inplace=True)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.relu(x)
        if self.up_scale and self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x


# ==============================
# 通道向量 <-> 2D 网格 (用于 2D FFT)
# ==============================
def vector_to_grid(x_vec):
    """
    x_vec: (B, C, 1, 1) -> (B, 1, Hc, Wc), 同时返回 (Hc, Wc, C, pad)
    使 C 尽量铺成接近方形的网格以做 2D FFT
    """
    B, C, _, _ = x_vec.shape
    Hc = int(math.floor(math.sqrt(C)))
    Wc = int(math.ceil(C / Hc))
    pad = Hc * Wc - C
    if pad > 0:
        x_vec = F.pad(x_vec.view(B, C), (0, pad))  # 在通道描述末尾补零
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


# ==============================
# 频率半径网格 (0~1 归一化)
# ==============================
def normalized_freq_radius(h, w, device=None, dtype=None):
    """
    生成频率半径 r \in [0,1]，基于 torch.fft.fftfreq
    r = sqrt(fx^2 + fy^2) / r_max
    """
    fy = torch.fft.fftfreq(h, d=1.0).to(device=device, dtype=dtype)  # [-0.5,0.5) 尺度
    fx = torch.fft.fftfreq(w, d=1.0).to(device=device, dtype=dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fx * fx + fy * fy)
    # 最大半径（Nyquist 对角）
    r_max = math.sqrt((0.5 ** 2) * 2.0)
    r = (r / r_max).clamp(0, 1)
    return r  # [H, W]


# ==============================
# ECA: 高效通道注意力（轻量1D卷积）
# ==============================
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: [B,C,H,W] -> [B,C,1,1] -> [B,1,C]
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)  # [B, C, 1] -> [B,1,C]
        y = self.conv1d(y).transpose(1, 2).unsqueeze(-1)             # [B,1,C] -> [B,C,1,1]
        return torch.sigmoid(y)


# ================================================
# FG-RCA: Fourier-Gated Residual Channel Attention
# ================================================
class FG_RCA(nn.Module):
    """
    🧠 模块名称：FG-RCA —— Fourier-Gated Residual Channel Attention
    设计要点：
    - Step1: GAP 得到通道描述向量
    - Step2: 向量 -> 2D 网格，做 FFT 分离振幅/相位
    - Step3: 对振幅/相位做非线性调制，并通过【频率门控】加强高频
    - Step4: IFFT -> 通道权重（Sigmoid）
    - Step5: 与 ECA 通道注意力融合，残差叠加；可选 DNRU 上采样
    """
    def __init__(
        self,
        channels: int,
        up_scale: int = 1,
        negative_slope: float = 0.1,
        eca_ksize: int = 3,
        freq_gate_init_t: float = 0.35,
        freq_gate_init_s: float = 8.0,
    ):
        super().__init__()
        self.channels = channels

        # 空间域轻量前处理（可学习更精细的低频/纹理前置特征）
        self.spatial_pre = CLC(channels, channels, k=3, negative_slope=negative_slope)

        # 频域 振幅/相位 轻量非线性映射
        act = nn.GELU()
        self.amp_mlp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            act,
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )
        self.pha_mlp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            act,
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )

        # 频率门控参数：阈值 t 与陡峭度 s（Sigmoid 门），以及高/低频整体缩放
        self.freq_t = nn.Parameter(torch.tensor(freq_gate_init_t, dtype=torch.float32))
        self.freq_s = nn.Parameter(torch.tensor(freq_gate_init_s, dtype=torch.float32))
        self.hi_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # 高频整体增益
        self.lo_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # 低频整体增益
        self.phs_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))  # 相位调制幅度

        # ECA 通道注意力
        self.eca = ECA(channels, k_size=eca_ksize)

        # 融合权重（可学习在频域与ECA之间的权衡）
        self.mix_alpha = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))  # in [0,1] 约束用 sigmoid
        self.mix_beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))   # 额外尺度

        # 轻量后处理 + 可选上采样
        self.post = DNRU(channels, up_scale=up_scale)

    @staticmethod
    def _sigm(x):
        return torch.sigmoid(x)

    def _freq_gate(self, h, w, device, dtype):
        """
        基于半径频率 r 得到门控:
        gate = sigmoid( s * (r - t) )
        -> 高频区域 gate ~ 1，低频区域 gate ~ 0
        最终频域缩放: lo_gain*(1-gate) + hi_gain*gate
        """
        r = normalized_freq_radius(h, w, device=device, dtype=dtype)  # [H,W]
        gate = torch.sigmoid(self.freq_s * (r - self.freq_t))         # [H,W]
        scale = self.lo_gain * (1.0 - gate) + self.hi_gain * gate
        return gate, scale  # 两种形式: gate 用于调制相位，scale 用于调制振幅

    def forward(self, x):
        """
        x: [B,C,H,W] -> 输出同形状
        """
        B, C, H, W = x.shape
        assert C == self.channels, "channels mismatch"

        # 空间域预处理
        feat = self.spatial_pre(x)  # [B,C,H,W]

        # Step1: GAP -> 通道描述向量
        chan_desc = F.adaptive_avg_pool2d(feat, 1)  # [B,C,1,1]

        # Step2: 向量->网格，FFT 分离振幅/相位
        grid, meta = vector_to_grid(chan_desc)                          # [B,1,Hc,Wc]
        spec = torch.fft.fft2(grid)                                     # complex [B,1,Hc,Wc]
        amp = torch.abs(spec)                                           # [B,1,Hc,Wc]
        pha = torch.angle(spec)                                         # [B,1,Hc,Wc]

        # 频率门控 (依据网格大小计算)
        _, Hc, Wc = spec.shape[1], spec.shape[2], spec.shape[3]
        gate, scale = self._freq_gate(Hc, Wc, grid.device, grid.dtype)  # [Hc,Wc]

        # Step3: 振幅/相位非线性 + 频域门控
        # 振幅放大: amp' = amp * (1 + amp_mlp(amp)) * scale
        amp_adj = amp * (1.0 + self.amp_mlp(amp)) * scale.unsqueeze(0).unsqueeze(0)
        # 相位微调: pha' = pha + phs_scale * gate * tanh(pha_mlp(pha))
        pha_adj = pha + self.phs_scale * gate.unsqueeze(0).unsqueeze(0) * torch.tanh(self.pha_mlp(pha))

        # Step4: 频谱重建 -> IFFT -> 通道权重
        spec_new = torch.polar(amp_adj, pha_adj)                        # complex
        grid_ifft = torch.fft.ifft2(spec_new).real                      # [B,1,Hc,Wc]
        weight_vec = grid_to_vector(grid_ifft, meta)                    # [B,C,1,1]
        w_freq = torch.sigmoid(weight_vec)                              # 频域得到的通道权重

        # ECA 通道注意力
        w_eca = self.eca(feat)                                          # [B,C,1,1]

        # Step5: 融合与残差
        alpha = torch.sigmoid(self.mix_alpha)                            # [0,1]
        w = self._sigm(self.mix_beta) * (alpha * w_freq + (1 - alpha) * w_eca)
        y = feat * w                                                    # 通道注意力作用
        out = y + x                                                     # 残差

        # 轻量后处理（可选上采样）
        out = self.post(out)
        return out


# ==============================
# 简单自测
# ==============================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W)
    module = FG_RCA(channels=C, up_scale=1)
    y = module(x)

    print(module)
    print("\n哔哩哔哩/微信公众号:CV缝合救星\n")
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
