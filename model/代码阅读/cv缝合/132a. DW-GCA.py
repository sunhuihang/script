import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================
# DW-GCA: Directional Wavelet Gated Cross-band Attention
# 方向性小波门控跨子带注意力（CVPR风格命名）
# 在原始 WaveletAttention 基础上加入：
# 1) 子带方向性门控（为 LH/HL/HH 单独学习门）
# 2) 跨子带注意力（在每个空间位置上，学习3个子带的加权分配）
# 3) 可学习子带再分配（将融合特征按通道再映射回 LH/HL/HH 以便 IDWT）
# 4) 混合通道注意力（SE + Softmax 归一化）与可学习残差融合
# =========================================

# =========================
# 构造 Daubechies-1 (Haar) 小波核
# =========================
def build_wavelet_kernels(device=None, dtype=torch.float32):
    """
    返回 2x2 的四个 2D 分析核：LL, LH, HL, HH
    对于 Db1 (Haar)：h0=[1/sqrt2, 1/sqrt2], h1=[-1/sqrt2, 1/sqrt2]
    2D 核是外积：h_row^T * h_col
    """
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      # 低通
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     # 高频
    # 外积得到 2x2 核
    LL = torch.ger(h0, h0)  # 低-低
    LH = torch.ger(h0, h1)  # 低-高（垂直边更敏感）
    HL = torch.ger(h1, h0)  # 高-低（水平边更敏感）
    HH = torch.ger(h1, h1)  # 高-高（对角边/角点）
    # 形状统一为 (4,1,2,2) 方便后续扩展到 groups=C
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
    return filt  # (4,1,2,2)


class SE(nn.Module):
    """
    轻量通道注意力（Squeeze-Excitation风格）
    - 使用 GAP 降维 -> 两层1x1线性 -> Sigmoid
    - 用于与 Softmax 通道归一化互补：SE 强调“绝对重要性”，Softmax 强调“相对分配”
    """
    def __init__(self, channels, r=8):
        super().__init__()
        hidden = max(channels // r, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B,C,H,W]
        z = F.adaptive_avg_pool2d(x, 1)         # [B,C,1,1]
        z = F.relu(self.fc1(z), inplace=True)
        z = torch.sigmoid(self.fc2(z))          # [B,C,1,1]
        return z


class DW_GCA(nn.Module):
    """
    DW-GCA：方向性小波门控跨子带注意力
    总流程：
    1) DWT: X -> (LH, HL, HH, LL)
    2) 高频软阈值（可学习阈值，按通道缩放到每个子带的能量尺度）
    3) 方向性门控：为 LH/HL/HH 学习到通道级别的 gate（Sigmoid）
    4) 跨子带注意力（Cross-band Attention）：
       - 将 LH/HL/HH reshape 为 [B,C,3,h,w]，对 dim=2 做 softmax，得到逐像素的子带权重
       - 对三个子带进行加权和，得到融合高频 H_mix（强调当前像素更重要的方向）
    5) 可学习子带再分配：
       - 对每个通道学习 w_LH, w_HL, w_HH（softmax到3个），将 H_mix 再分配回 LH2/HL2/HH2 以便 IDWT
    6) IDWT：用 (LH2, HL2, HH2, LL) 重建 X_re
    7) 通道注意力（SE） + Softmax 通道归一化 的混合权重
    8) 残差融合： out = x * w_softmax + gamma * X_re * w_se
       gamma 为可学习标量，控制重建分支对最终输出的贡献
    """
    def __init__(self, channels, use_fc=True, se_ratio=8):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc

        # ---------- 高频软阈值参数（3个子带 * C） ----------
        # 用 Sigmoid 将其约束到 [0,1]，再自适应缩放到子带的均值幅度
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # ---------- 方向性门控（3个子带 * C），Sigmoid ----------
        # 用于在阈值后对不同方向的响应进行通道级再加权
        self.dir_gate = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # ---------- 可选的 FC（对 GAP 后的通道向量做线性变换，便于Softmax通道归一化前的可学习投影） ----------
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)

        # ---------- 可学习的子带再分配系数（C x 3），用于将融合后的 H_mix 映射回 LH/HL/HH ----------
        # 用 softmax 保证三者之和为1，提高可解释性与稳定性
        self.sub_redistribute = nn.Parameter(torch.zeros(channels, 3))  # 初始化为0，softmax后约等于均分

        # ---------- 通道注意力（SE） ----------
        self.se = SE(channels, r=se_ratio)

        # ---------- 可学习残差系数 ----------
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # ---------- 小波核（注册为 buffer，不参与训练） ----------
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt)  # Haar 正交：合成=分析

    # ---------- DWT 与 IDWT ----------
    def dwt(self, x):
        """
        x: (B,C,H,W)
        返回：LH, HL, HH, LL
        """
        B, C, H, W = x.shape

        # 填充到偶数尺寸，保证stride=2整除（避免边界丢失）
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # 组卷积：每个通道使用同一组 4 个滤波器
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)

        # 按子带拆分： [LL, LH, HL, HH] 顺序与上面 build 函数保持一致
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()
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

        # 反卷积作为合成滤波器，stride=2
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return x_rec

    # ---------- 高频软阈值 ----------
    @staticmethod
    def soft_threshold(x, thr):
        # 经典 soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        """
        x: [B,C,H,W]
        输出：与输入同形状 [B,C,H,W]
        """
        B, C, H, W = x.shape

        # 1) 小波分解
        LH, HL, HH, LL = self.dwt(x)

        # 2) 高频子带软阈值（自适应按通道尺度化阈值）
        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)  # (3,C,1,1), 映射到 0~1
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        # 3) 方向性门控（通道级）：为 LH/HL/HH 引入 Sigmoid 门
        g = torch.sigmoid(self.dir_gate)  # (3,C,1,1)
        LH_hat = LH_hat * g[0].unsqueeze(0)   # 加强/抑制对“垂直边”更敏感的通道
        HL_hat = HL_hat * g[1].unsqueeze(0)   # 加强/抑制对“水平边”更敏感的通道
        HH_hat = HH_hat * g[2].unsqueeze(0)   # 加强/抑制对“对角/角点”更敏感的通道

        # 4) 跨子带注意力（逐像素地决定更信任哪个方向）
        # 将三个子带堆成 [B, C, 3, h, w]，对 dim=2 做 softmax，得到跨子带权重
        h = LH_hat.size(-2)
        w = LH_hat.size(-1)
        stack3 = torch.stack([LH_hat, HL_hat, HH_hat], dim=2)  # [B,C,3,h,w]
        # a: [B,C,3,h,w] -> softmax over band-dimension
        attn_band = F.softmax(stack3.abs().mean(dim=1, keepdim=True), dim=2)  # 用跨通道的能量引导（更稳定）
        # 也可以直接对 stack3 做一个 1x1x1 的线性映射后 softmax，这里用能量引导更轻量

        # 按权重加权求和得到融合高频 H_mix（仍为 [B,C,h,w]）
        H_mix = (stack3 * attn_band).sum(dim=2)  # [B,C,h,w]

        # 5) 可学习子带再分配：将 H_mix 映射回 LH2/HL2/HH2，保证可以做 IDWT
        # 对每个通道有3个权重，softmax 到 3 个子带
        w_redis = F.softmax(self.sub_redistribute, dim=1)  # [C,3]
        # reshape 为 [1,C,3,1,1] 便于广播
        w_redis = w_redis.view(1, C, 3, 1, 1)
        # 将 H_mix 拓展成3份，然后乘以通道的再分配系数
        H_mix_exp = H_mix.unsqueeze(2)  # [B,C,1,h,w]
        H_redist = H_mix_exp * w_redis  # [B,C,3,h,w]
        # 拆回三路
        LH2 = H_redist[:, :, 0]
        HL2 = H_redist[:, :, 1]
        HH2 = H_redist[:, :, 2]

        # 6) 逆小波重建，得到重建特征 X_re
        X_re = self.idwt(LH2, HL2, HH2, LL)  # [B,C,H',W'] 与输入 H/W 基本一致

        # 7) 通道注意力：SE（绝对重要性） + Softmax（相对分配）
        # 7.1 Softmax 通道归一化（可选FC增强可分性）
        gap_vec = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # [B,C]
        if self.use_fc:
            gap_vec = self.fc(gap_vec)                       # [B,C]
        w_softmax = F.softmax(gap_vec, dim=1).view(B, C, 1, 1)  # [B,C,1,1]

        # 7.2 SE 通道权重（Sigmoid）
        w_se = self.se(X_re)  # [B,C,1,1]

        # 8) 残差融合：输入经 softmax 权重强调（更“分配式”），重建分支经 SE 强调（更“绝对式”）
        out = x * w_softmax + self.gamma * (X_re * w_se)
        return out


# ==============================
# 简单自测
# ==============================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 1, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    # 新模块：DW-GCA（方向性小波门控跨子带注意力）
    module = DW_GCA(channels=C, use_fc=True, se_ratio=8).to(device)
    y = module(x)

    print(module)
    print("\n模块名称：DW-GCA（Directional Wavelet Gated Cross-band Attention）")
    print("说明：方向性门控 + 跨子带注意力 + 子带再分配 + SE 与 Softmax 混合通道注意力 + 可学习残差")
    print(f"\nInput  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)}")
