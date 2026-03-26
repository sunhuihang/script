import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_disjoint_fourier_masks(k: int, cin: int, cout: int, n_groups: int, device):
    """
    构造频率域的互不相交索引掩码（从低频到高频均匀分桶）。
    我们把 (k*Cin, k*Cout) 视作二维频谱网格，对频率半径排序后均分成 n_groups 份。
    """
    H, W = k * cin, k * cout  # “频谱平面”的尺寸
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    # 半径（以中心为原点）
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # 将半径展开排序，按分位点分桶
    flat = rr.flatten()
    quantiles = torch.quantile(flat, torch.linspace(0, 1, steps=n_groups + 1, device=device))
    masks = []
    for gi in range(n_groups):
        low, high = quantiles[gi], quantiles[gi + 1] + 1e-6
        m = (rr >= low) & (rr < high)
        masks.append(m.to(torch.bool))
    return masks  # list[BoolTensor] with shape [k*Cin, k*Cout]


class PAFDC(nn.Module):
    """
    PAF-DC: Phase-Aware Fourier-Disjoint Dynamic Convolution
    - 频率不相交权重（FDW）：固定参数预算，在频率域将谱系数分到 n 个不相交频带，iFFT 得到 n 组核权重。
    - 相位调制（Phase-aware）：由输入特征预测每个频段的相位偏移，对应地旋转复谱系数 -> 响应对输入更自适应。
    - 频带门控（Band gating）：对 n 个频段对应的卷积输出，预测空间位置相关 gate 并融合。
    - 轻量空间核调制（Lite-KSM）：用局部+全局门控对最终核做逐元素的轻量调制（避免巨大参数量）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
        n_bands: int = 4,          # 频段/不相交组数
        phase_hidden: int = 32,    # 相位预测 MLP 隐藏维
        band_gate_kernel: int = 3, # 频带门控的空间卷积感受野
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel for best effect."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.n_bands = n_bands

        # -------- 1) 共享的频域参数（幅度），固定参数预算 --------
        # 我们学习一个实数谱幅度矩阵 A（非负通过 softplus），大小为 (k*Cin, k*Cout)
        self.A = nn.Parameter(torch.randn(self.k * in_channels, self.k * out_channels) * 0.02)
        self.A_act = nn.Softplus()  # 保证非负，稳定

        # 每个频段一个可学习的幅度缩放（更易训练）
        self.band_scale = nn.Parameter(torch.ones(n_bands))

        # -------- 2) 相位调制：由输入预测 n_bands 个相位偏移 φ_b(x) --------
        # 使用 GAP 后的 MLP 预测 (B, n_bands)
        self.phase_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, phase_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(phase_hidden, n_bands),
        )

        # -------- 3) 轻量空间核调制（Lite-KSM）--------
        # 通道层面：输入/输出通道门控（SE风格）
        self.in_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(8, in_channels // 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 16), in_channels, 1),
            nn.Sigmoid(),
        )
        self.out_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(8, out_channels // 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, out_channels // 16), out_channels, 1),
            nn.Sigmoid(),
        )
        # 空间核维度门控：生成 (k,k) 的核空间门控
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.k * self.k, 1),
            nn.Sigmoid(),  # 映射到 0~1
        )

        # -------- 4) 频带门控：对每个频带输出生成 (H,W) 的门控 --------
        # 用轻量 conv 预测 n_bands 张 gate map
        pad = band_gate_kernel // 2
        self.band_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, band_gate_kernel, padding=pad, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, n_bands, 1),
            nn.Sigmoid(),
        )

        # 最终 1x1 融合（可选）：整合多频带输出后再线性变换
        self.fuse = nn.Conv2d(out_channels, out_channels, 1, bias=True) if out_channels > 1 else nn.Identity()

        # 标准卷积偏置
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # 运行时缓存的频率掩码（按设备构建）
        self._cached = {}

    def _get_masks(self, device):
        key = (device, self.k, self.in_channels, self.out_channels, self.n_bands)
        if key in self._cached:
            return self._cached[key]
        masks = _build_disjoint_fourier_masks(self.k, self.in_channels, self.out_channels, self.n_bands, device)
        # 打包成 (n_bands, k*Cin, k*Cout) 的 bool tensor
        M = torch.stack(masks, dim=0)
        self._cached[key] = M
        return M

    def _ifft_to_weights(self, A_mag: torch.Tensor, phase: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        从幅度 A_mag (kCin, kCout)、相位向量 phase (B, n_bands) 和掩码 masks (n_bands, kCin, kCout)
        生成 batch-wise 的 n_bands 组空间核：
        返回形状: (B, n_bands, k, k, Cin, Cout)
        """
        B = phase.shape[0]
        kCin, kCout = A_mag.shape
        device = A_mag.device

        # 把 A_mag 按频段 mask 分配，并分别乘以输入相关的相位项 e^{j*phi_b}
        # 最终构造复谱：A_b * exp(j*phi_b)
        A_bands = []
        for b in range(self.n_bands):
            Mb = masks[b]  # bool mask
            Ab = torch.zeros_like(A_mag, dtype=torch.complex64, device=device)
            # 按band缩放幅度
            mag = self.band_scale[b] * A_mag
            # 相位旋转（batch化）
            # 这里将相位扩展成 (B, 1, 1)，便于广播到复谱矩阵
            ph = phase[:, b].view(B, 1, 1)
            complex_phase = torch.exp(1j * ph).to(torch.complex64)
            # 只在该 band 的索引处写入复数值，其他为 0
            # 先把幅度放到实部，再乘以相位
            base = mag.masked_fill(~Mb, 0.0).to(torch.complex64)
            Ab_complex = base[None, :, :] * complex_phase  # (B, kCin, kCout)
            A_bands.append(Ab_complex)

        # IFFT 到空间：我们将 (kCin, kCout) 视作 (k*Cin, k*Cout) 的 2D 频谱
        # torch 内置的 ifft2 作用在最后两个维度
        weights_per_band = []
        for Ab in A_bands:
            # (B, kCin, kCout) -> ifft2 -> 取实部作为空间域
            spatial = torch.fft.ifft2(Ab, norm="ortho").real  # (B, kCin, kCout)
            # 切成 (k, Cin) × (k, Cout) 的网格块，再重组为 (k, k, Cin, Cout)
            Bsz = spatial.shape[0]
            # 先 reshape 成 (B, Cin, k, Cout, k)，再 permute 为 (B, k, k, Cin, Cout)
            spatial = spatial.view(Bsz, self.in_channels, self.k, self.out_channels, self.k)
            spatial = spatial.permute(0, 2, 4, 1, 3).contiguous()  # (B, k, k, Cin, Cout)
            weights_per_band.append(spatial)

        # 堆叠：(B, n_bands, k, k, Cin, Cout)
        W = torch.stack(weights_per_band, dim=1)
        return W

    def _apply_lite_ksm(self, W, x):
        """
        对核进行轻量空间调制：
        - in_gate: (B, Cin, 1, 1)
        - out_gate: (B, Cout, 1, 1)
        - spatial_gate: (B, k*k, H, W) -> 我们只取空间平均，得到 (B, k, k, 1, 1) 的核空间门控
        最终将三者相乘到 W 上（逐元素广播）。
        """
        B = x.shape[0]
        in_g = self.in_gate(x)           # (B, Cin, 1, 1)
        out_g = self.out_gate(x)         # (B, Cout, 1, 1)

        sg = self.spatial_gate(x)        # (B, k*k, H, W)
        # 取空间平均，得到核空间维度门控
        sg = sg.mean(dim=[2, 3])         # (B, k*k)
        sg = sg.view(B, self.k, self.k, 1, 1)  # (B, k, k, 1, 1)

        # 将 in/out gate broadcast 到核形状
        in_g = in_g.view(B, 1, 1, self.in_channels, 1)    # (B,1,1,Cin,1)
        out_g = out_g.view(B, 1, 1, 1, self.out_channels) # (B,1,1,1,Cout)

        # W: (B, n_bands, k, k, Cin, Cout)
        W = W * sg[:, None] * in_g[:, None] * out_g[:, None]
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, H, W)
        返回: (B, Cout, H_out, W_out)
        """
        B, C, H, W = x.shape
        device = x.device

        # 1) 频域幅度（非负）
        A_mag = self.A_act(self.A)  # (kCin, kCout)

        # 2) 获取 disjoint masks
        masks = self._get_masks(device)  # (n_bands, kCin, kCout)

        # 3) 由输入预测每个 band 的相位偏移 (B, n_bands)
        phase = self.phase_mlp(x)  # (B, n_bands)

        # 4) IFFT -> 得到 batch-wise 的频带核 (B, n_bands, k, k, Cin, Cout)
        W_bands = self._ifft_to_weights(A_mag, phase, masks)

        # 5) 轻量 KSM：对核逐元素调制
        W_bands = self._apply_lite_ksm(W_bands, x)  # (B, n_bands, k, k, Cin, Cout)

        # 6) 用每个频带的核分别做“按样本”的卷积
        # 为避免自写按样本卷积，这里采用分组技巧：把 batch 合到 group 维
        # 先将权重 reshape 为 (B*n_bands*Cout, Cin, k, k)，输入扩展为 (1, B*Cin, H, W) 并做 grouped conv
        W_agg = W_bands.permute(0, 1, 5, 4, 2, 3).contiguous()  # (B, n_bands, Cout, Cin, k, k)
        W_agg = W_agg.view(B * self.n_bands * self.out_channels, self.in_channels, self.k, self.k)

        x_exp = x.view(1, B * C, H, W)
        y = F.conv2d(
            x_exp,
            W_agg,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B,  # 关键：将 batch 维作为 groups
        )
        # 现在 y 形状为 (1, B*n_bands*Cout, H_out, W_out) -> 还原
        Hout, Wout = y.shape[-2], y.shape[-1]
        y = y.view(B, self.n_bands, self.out_channels, Hout, Wout)

        # 7) 频带门控：预测 (B, n_bands, H_out, W_out)
        gates = self.band_gate(F.interpolate(x, size=(Hout, Wout), mode="bilinear", align_corners=False))
        # 加权融合
        y = (y * gates.unsqueeze(2)).sum(dim=1)  # (B, Cout, H_out, W_out)

        # 8) 线性融合与偏置
        y = self.fuse(y)
        if self.use_bias and self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        return y


# ---------------------- Demo ---------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, Cin, H, W = 2, 32, 64, 64
    Cout = 32

    x = torch.randn(B, Cin, H, W, device=device)

    pafdc = PAFDC(
        in_channels=Cin,
        out_channels=Cout,
        kernel_size=3,
        stride=1,
        padding=1,
        n_bands=4,           # 频带数（可调大一些提升频率分辨率）
        phase_hidden=32,     # 相位预测 MLP 隐藏维
        band_gate_kernel=3,  # 频带门控的卷积核
    ).to(device)

    with torch.no_grad():
        y = pafdc(x)
    print(pafdc)
    print("Input :", x.shape)
    print("\n微信公众号:CV缝合救星！\n")
    print("Output:", y.shape)  # 期望 (B, Cout, H, W)
