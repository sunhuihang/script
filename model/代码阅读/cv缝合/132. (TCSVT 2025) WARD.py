import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 哔哩哔哩 / 微信公众号： CV缝合救星 独家复现
# =========================
# 构造 Daubechies-1 (Haar) 小波核
# =========================
def build_wavelet_kernels(device=None, dtype=torch.float32):                                                                                                                                                                     # 微信公众号:CV缝合救星
    """
    返回 2x2 的四个 2D 分析核：LL, LH, HL, HH
    对于 Db1 (Haar)：h0=[1/sqrt2, 1/sqrt2], h1=[-1/sqrt2, 1/sqrt2]                                                                                                                                                                     # 微信公众号:CV缝合救星
    2D 核是外积：h_row^T * h_col
    """
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      # 低通                                                                                                                                                                     # 微信公众号:CV缝合救星
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     # 高频                                                                                                                                                                     # 微信公众号:CV缝合救星
    # 外积得到 2x2 核
    LL = torch.ger(h0, h0)  # 低-低
    LH = torch.ger(h0, h1)  # 低-高（垂直边）
    HL = torch.ger(h1, h0)  # 高-低（水平边）
    HH = torch.ger(h1, h1)  # 高-高（对角）
    # 形状统一为 (1,1,2,2) 方便后续扩展到 groups=C
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)                                                                                                                                                                     # 微信公众号:CV缝合救星
    return filt  # (4,1,2,2)


# =========================
# Wavelet Attention 模块
# =========================
class WaveletAttention(nn.Module):
    """
    实现步骤：
    X --DWT--> (LH, HL, HH, LL)
         高频阈值化 -> concat -> 1x1 conv 融合 -> 与 LL 做 IDWT -> X_re
         GAP -> (可选FC) -> Softmax -> 通道权重
         输出： Final = weight * X
    """
    def __init__(self, channels, use_fc=True):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc

        # 软阈值参数（3 个高频子带 * C），sigmoid 约束到 0~1，再乘以 mean(|x|)
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # 高频子带融合：将 3C -> C
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)                                                                                                                                                                     # 微信公众号:CV缝合救星

        # GAP 后可选的 FC（保持维度 C->C）
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)

        # 小波核（注册为 buffer，参与 to(device) 但不训练）
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt)  # Db1 正交：合成=分析

    # ---------- DWT 与 IDWT ----------
    def dwt(self, x):
        """
        x: (B,C,H,W)
        返回：LH, HL, HH, LL 以及中间 size 信息
        """
        B, C, H, W = x.shape

        # 零填充到偶数尺寸，避免边界丢失
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)                                                                                                                                                                     # 微信公众号:CV缝合救星

        # 组卷积：每个通道使用同一组 4 个滤波器
        # 权重形状需要扩展为 (4*C, 1, 2, 2) 并 groups=C
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)                                                                                                                                                                     # 微信公众号:CV缝合救星

        # 按子带拆分
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()                                                                                                                                                                     # 微信公众号:CV缝合救星
        LL = y[:, :, 0]  # (B,C,h,w)
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        """
        逆变换：将四个子带重建为 (B,C,H,W)
        """
        B, C, h, w = LL.shape
        # 将 4 个子带 stack 回 (B,4C,h,w)
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)

        # conv_transpose2d 作为合成滤波器，stride=2
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        # conv_transpose 的权重形状：(in_channels, out_channels/groups, kH, kW)
        # 我们希望 groups=C，每组把 4 个子带合成为 1 个通道
        # 需要把 weight 视作 (4C, 1, 2, 2)，设置 groups=C 时会自动每4个输入映射到1个输出
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)                                                                                                                                                                     # 微信公众号:CV缝合救星
        return x_rec

    # ---------- 高频软阈值 ----------
    @staticmethod
    def soft_threshold(x, thr):
        # soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    # ---------- 前向 ----------
    def forward(self, x):
        B, C, H, W = x.shape

        # 1) DWT
        LH, HL, HH, LL = self.dwt(x)

        # 2) 高频子带阈值化与融合
        # 归一化后的阈值（按通道），值域约束 0~1，再乘以该子带的平均幅度
        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)  # (3,C,1,1)
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        # 融合卷积（将 3C -> C）
        H_concat = torch.cat([LH_hat, HL_hat, HH_hat], dim=1)  # (B,3C,h,w)                                                                                                                                                                     # 微信公众号:CV缝合救星
        H_fused = self.fuse(H_concat)  # (B,C,h,w)

        # 3) IDWT 重构
        X_re = self.idwt(LH_hat, HL_hat, H_fused, LL)  # (B,C,H',W')，H'/W'≈H/W                                                                                                                                                                     # 微信公众号:CV缝合救星

        # 4) 注意力权重：GAP -> (可选FC) -> Softmax(沿通道)
        gap = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # (B,C)
        if self.use_fc:
            gap = self.fc(gap)  # (B,C)
        attn = F.softmax(gap, dim=1).view(B, C, 1, 1)  # (B,C,1,1)

        # 5) 加权原输入
        out = x * attn
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"                                                                                                                                                                     # 微信公众号:CV缝合救星

    B, C, H, W = 1, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    wa = WaveletAttention(channels=C, use_fc=True).to(device)                                                                                                                                                                     # 微信公众号:CV缝合救星
    y= wa(x)

    print(wa)
    print("\n哔哩哔哩/微信公众号:CV缝合救星, 独家复现! \n")
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)}")