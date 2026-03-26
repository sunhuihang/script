# -*- coding: utf-8 -*-
"""
CoDA: Cross-gated Dual-domain Adaptive Coordinate Attention
跨门控双域自适应坐标注意力
作者： CV缝合救星
版权：微信公众号：CV缝合救星
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoDA(nn.Module):
    """
    CoDA（Cross-gated Dual-domain Adaptive Coordinate Attention）
    设计动机：
      1）坐标注意力对 H/W 方向进行解耦建模，但对不同噪声类型/强度的自适应性有限；
      2）仅依赖空间域统计，难以感知频域能量分布中与噪声相关的高频成分；
      3）H/W 两个分支彼此独立，缺乏跨方向的相互约束与选择。

    关键创新（相对原始 ACA 的“魔改”）：
      A. 频域门控（Frequency Gate）：使用 torch.fft.fft2 获取幅度谱，学习得到逐通道门控，
         在不显著增加开销的前提下引入“频域先验”，提升对不同噪声形态的适配性。
      B. 方向长程建模（Directional DW-Conv）：在 H/W 两个分支上加入深度可分一维卷积
         （k×1 与 1×k），抓取跨行/跨列的长程依赖，增强细粒度噪声的捕捉。
      C. 跨门控融合（Cross-gating）：使用对向分支的全局门信号对本分支进行调制，
         让 H/W 两个方向“相互选择、相互抑制”，避免冗余响应。
      D. 可学习温度与层缩放（Temperature & LayerScale）：为 Sigmoid 引入温度 τ（可学习），
         并对残差分支加入 γ 的层缩放，稳定训练、便于与现有骨干网即插即用。

    备注：保留了 ACA 的自适应缩放思想（alpha），并保持轻量化与即插即用特性。
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        alpha: float = 0.9,
        kernel_size: int = 7,
        use_frequency_gate: bool = True,
        layerscale_init: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 需为奇数，便于对齐（padding 对称）"
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)  # 瓶颈通道数
        self.alpha = alpha
        self.use_frequency_gate = use_frequency_gate

        # 共享 MLP：对 H/W 拼接后的方向统计做通道压缩与非线性映射
        # 微信公众号：CV缝合救星
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        # 方向长程建模：深度可分一维卷积（沿 H 与 W）
        # 目的：对每个方向引入更强的跨行/跨列上下文
        pad = kernel_size // 2
        self.dw_h = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=self.mid_channels,
            bias=False,
        )
        self.dw_w = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            groups=self.mid_channels,
            bias=False,
        )

        # 方向映射回原通道维度
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False)

        # 频域门控：逐通道门（0~1），由幅度谱的全局统计产生
        if self.use_frequency_gate:
            self.freq_gate = nn.Sequential(
                nn.Conv2d(in_channels, max(8, in_channels // reduction), kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(8, in_channels // reduction), in_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        # 可学习温度，用于调节 Sigmoid 的锐度（τ 越小越“硬”）
        self.tau = nn.Parameter(torch.tensor(1.0))

        # 层缩放参数（LayerScale），稳定深层残差叠加
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * layerscale_init)

        # 轻微随机失活，进一步抑制过拟合（可选）
        self.dropout = nn.Dropout(p=0.05)

    def _sigmoid_temp(self, x):
        # 带温度的 Sigmoid：sigmoid(x / tau)
        return torch.sigmoid(x / (self.tau.abs() + 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, C, H, W]
        输出：
            y: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # -------- 1) 方向统计（与 ACA 一致的骨架）--------
        x_h = F.adaptive_avg_pool2d(x, (H, 1))  # [B, C, H, 1] —— 沿宽度聚合，保留行方向
        x_w = F.adaptive_avg_pool2d(x, (1, W))  # [B, C, 1, W] —— 沿高度聚合，保留列方向
        x_w = x_w.permute(0, 1, 3, 2)          # 变成 [B, C, W, 1]，便于与 H 方向拼接

        # 拼接后用共享 MLP 抽取方向相关的通道表征
        y = torch.cat([x_h, x_w], dim=2)       # [B, C, H+W, 1]
        y = self.shared_conv(y)                # [B, C', H+W, 1]，C' = mid_channels

        # 切分回两个方向分支，并做“方向深度可分一维卷积”的长程建模
        y_h, y_w = torch.split(y, [H, W], dim=2)   # y_h:[B,C',H,1], y_w:[B,C',W,1]
        y_w = y_w.permute(0, 1, 3, 2)              # y_w:[B,C',1,W]

        y_h = self.dw_h(y_h)                       # 在 H 方向做 k×1 的 DWConv
        y_w = self.dw_w(y_w)                       # 在 W 方向做 1×k 的 DWConv

        # 映射回原通道数，并应用自适应缩放 alpha（延续 ACA 思想）
        a_h_raw = self.conv_h(y_h) * self.alpha     # [B,C,H,1]
        a_w_raw = self.conv_w(y_w) * self.alpha     # [B,C,1,W]

        # -------- 2) 跨门控融合（Cross-gating）--------
        # 使用对向分支的全局门对本分支进行调制，增强“相互选择/抑制”
        gate_h_from_w = a_w_raw.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        gate_w_from_h = a_h_raw.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]

        a_h = self._sigmoid_temp(a_h_raw) * (1.0 + gate_h_from_w)   # [B,C,H,1]
        a_w = self._sigmoid_temp(a_w_raw) * (1.0 + gate_w_from_h)   # [B,C,1,W]

        # -------- 3) 频域门控（Dual-domain 之二：Frequency）--------
        if self.use_frequency_gate:
            # 计算幅度谱的均值特征（逐通道）
            # 注：仅用于产生门控标量，不做反变换，计算量可控
            Xf = torch.fft.fft2(x, norm='ortho')            # 复数张量
            mag = torch.abs(Xf)                             # 幅度谱 [B,C,H,W]
            mag_mean = mag.mean(dim=(2, 3), keepdim=True)   # [B,C,1,1]
            f_gate = self.freq_gate(mag_mean)               # [B,C,1,1] in [0,1]
        else:
            f_gate = 0.0

        # -------- 4) 融合与输出 --------
        # 方向注意力相加（与坐标注意力一致的融合策略），再乘以频域门控（1 + f_gate）
        attn = a_h + a_w                                   # [B,C,H,W]（自动广播）
        attn = attn * (1.0 + f_gate)                       # 双域融合：空间方向 + 频域门控
        attn = self.dropout(attn)

        # 残差式输出，LayerScale 提升稳定性
        out = x + self.gamma * (x * attn)
        return out


# ===================== 测试脚本 ===================== #
if __name__ == "__main__":
    # 微信公众号：CV缝合救星
    torch.manual_seed(0)

    x = torch.randn(2, 32, 64, 64)  # batch=2, C=32, H=W=64
    model = CoDA(
        in_channels=32,
        reduction=16,
        alpha=0.9,
        kernel_size=7,             # 可调的方向一维卷积核
        use_frequency_gate=True,   # 开/关频域门控
        layerscale_init=0.1,
    )

    print(model)
    print("\n微信公众号:CV缝合救星\n")

    y = model(x)
    print("输入:", x.shape)
    print("输出:", y.shape)

    # 简单数值检查：梯度传播
    y.mean().backward()
    # 读取几个关键参数，确保可学习
    print("tau(温度):", float(model.tau.detach()))
    print("gamma(层缩放)均值:", float(model.gamma.mean().detach()))
