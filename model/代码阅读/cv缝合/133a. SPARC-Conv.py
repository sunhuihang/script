# -*- coding: utf-8 -*-
# ============================================================
# CV缝合救星
# SPARC-Conv (Self-Padded Adaptive Rotation-Consistent Convolution)
# 主要创新点（与原 PreCM 相比）：
# 1）自适应方向融合（Orientation Attention）：根据输入内容动态估计4个方向分支的融合权重（Softmax），
#    让网络在不同图像/场景下自动选择更有信息量的方向分支，提升旋转鲁棒性与表达力。
# 2）双模式自适应填充（Dual-Pad Gating）：每个方向分支内部引入“零填充 vs 复制填充”的可学习门控（Sigmoid），
#    动态平衡边界伪影与上下文保留，缓解旋转+卷积带来的边界不一致问题。
# 3）轻量高频强调（Edge-aware Boosting）：利用 Sobel 提取的边缘响应作为引导，对分支输出进行逐像素自适应增强，
#    使旋转等变下的细节边缘更清晰稳定。
#
# 该模块仍遵循 PreCM 的“输入/权重/填充 同步旋转 + 输出逆旋转对齐”的核心原则，
# ============================================================

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- 工具函数区 -----------------------------
def _rotate_padding_vector(p: Tuple[int, int, int, int], k: int) -> Tuple[int, int, int, int]:
    """
    旋转 padding 向量的置换规则。
    输入/返回格式均为 (pa, pl, pb, pr) = (上、左、下、右)。
    每顺时针旋转90°，根据论文化简后的置换：(a, l, b, r) -> (r, a, l, b)。
    连续应用 k 次。
    """
    pa, pl, pb, pr = p
    for _ in range(k % 4):
        pa, pl, pb, pr = pr, pa, pl, pb
    return pa, pl, pb, pr


def _compute_padding(h_in: int, w_in: int,
                     k: int, s: int, d: int,
                     h_out: int, w_out: int) -> Tuple[int, int, int, int]:
    """
    基于 PreCM 公式计算 σ0 的基础 padding，返回 (pa, pl, pb, pr) = (上、左、下、右)。
    注意：为了保证可运行性，若出现负 padding（严格意义上对应裁剪），此处将其截断为0。
    如需严格裁剪，可在卷积后手动裁剪到目标尺寸。
    """
    pab = (h_out - 1) * s + d * (k - 1) + 1 - h_in
    prl = (w_out - 1) * s + d * (k - 1) + 1 - w_in
    pb = pab // 2
    pl = prl // 2
    pa = pab - pb
    pr = prl - pl
    # 负值置零（工程可运行考虑）
    pa = int(max(pa, 0)); pb = int(max(pb, 0))
    pl = int(max(pl, 0)); pr = int(max(pr, 0))
    return pa, pl, pb, pr


# ----------------------------- 模块定义区 -----------------------------
class SPARCConv(nn.Module):
    """
    SPARC-Conv：在 PreCM 上的“魔改创新”版本（CVPR风格）
    - 4 个方向分支：0°/90°/180°/270°
      * 输入x、卷积核w、padding 同步旋转
      * 卷积输出再逆旋转对齐回参考方向
    - 自适应方向融合：基于全局语义自适应生成4个方向的融合权重（Softmax）。
    - 双模式自适应填充：每个方向分支对 zero / replicate 两种填充进行门控混合（Sigmoid）。
    - 高频强调：基于 Sobel 边缘响应对分支输出进行逐像素增强。
    - 输出尺寸与通道数与常规Conv一致（batch不变；通道=out_channels；空间尺寸与 output_shape 一致）。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 use_edge_boost: bool = True,
                 ):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0, "groups 必须整除通道数"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.use_edge_boost = use_edge_boost

        # （1）基础卷积核参数（σ0方向）
        w = torch.empty(out_channels, in_channels // groups, self.kernel_size, self.kernel_size)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight0 = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # （2）方向注意力（生成 4 个方向分支的融合权重）
        #     输入：全局池化后的 (B, C)；输出：每个样本的 4 维 logits -> softmax
        self.dir_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B,C,1,1)
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 4, 1, bias=True)  # (B,4,1,1)
        )

        # （3）填充模式门控（为每个方向生成 zero/replicate 的混合系数）
        #     同样基于全局池化，输出 4 个方向分支的 gate，Sigmoid ∈ (0,1)
        self.pad_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 4, 1, bias=True),  # (B,4,1,1)
            nn.Sigmoid()
        )

        # （4）高频强调：Sobel 边缘 + 1x1 卷积，产生与输出通道一致的调制掩码（Sigmoid）
        if self.use_edge_boost:
            # 固定的 Sobel 卷积核（作用于灰度图）
            sobel_x = torch.tensor([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
            # 将边缘强度映射到通道维度，并生成 [0,1] 的掩码
            self.edge_proj = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)
            self.edge_bn = nn.BatchNorm2d(out_channels)
            self.edge_act = nn.Sigmoid()
            # 高频增强强度（可学习标量，初始较小，避免不稳定）
            self.gamma = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    # ----------------------------- 前向主流程 -----------------------------
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        output_shape: (H_out, W_out)；若不传则默认 H_out=H, W_out=W。
        """
        B, C, H, W = x.shape
        if output_shape is None:
            h_out, w_out = H, W
        else:
            h_out, w_out = int(output_shape[0]), int(output_shape[1])

        # σ0 的基础 padding（后续对每个方向进行旋转置换）
        pa0, pl0, pb0, pr0 = _compute_padding(
            H, W, self.kernel_size, self.stride, self.dilation, h_out, w_out
        )

        # 方向注意力权重（B,4,1,1） -> softmax，保证四方向权重和为1
        dir_logits = self.dir_attn(x)        # (B,4,1,1)
        dir_weights = F.softmax(dir_logits, dim=1)  # (B,4,1,1)

        # 填充模式门控（B,4,1,1），越靠近1越偏向“复制填充 replicate”，越靠近0越偏向“零填充 zero”
        gate = self.pad_gate(x)  # (B,4,1,1)

        # 构建4个方向分支
        y_sum = None
        for k in range(4):  # 0,1,2,3 -> 0°,90°,180°,270°
            # 1) 旋转输入
            xk = torch.rot90(x, k=k, dims=(2, 3))

            # 2) 旋转 padding，并准备两种填充模式
            pak, plk, pbk, prk = _rotate_padding_vector((pa0, pl0, pb0, pr0), k)
            pad_tuple = (plk, prk, pak, pbk)  # F.pad 的顺序：(left,right,top,bottom)

            # 2.1) zero pad 路径
            xk_zero = F.pad(xk, pad_tuple, mode="constant", value=0.0) if any(pad_tuple) else xk
            # 2.2) replicate pad 路径
            xk_rep = F.pad(xk, pad_tuple, mode="replicate") if any(pad_tuple) else xk

            # 混合两种填充（逐样本、逐方向 gate）
            # gate[:,k] 形状 (B,1,1)；这里用广播构建到 (B,1,H,W)
            gk = gate[:, k:k+1, :, :]  # (B,1,1,1)
            xk_pad = gk * xk_rep + (1.0 - gk) * xk_zero

            # 3) 旋转卷积核权重
            wk = torch.rot90(self.weight0, k=k, dims=(2, 3))

            # 4) 方向分支卷积
            yk = F.conv2d(xk_pad, weight=wk, bias=self.bias,
                          stride=self.stride, dilation=self.dilation, groups=self.groups)

            # 5) 逆旋转对齐回参考方向
            yk = torch.rot90(yk, k=(4 - k) % 4, dims=(2, 3))  # (B, C_out, H_out, W_out)

            # 6) 可选：高频强调（边缘引导逐像素增强）
            if self.use_edge_boost:
                # 将原始输入转为“灰度”近似：通道平均
                x_gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
                # Sobel 提取梯度
                gx = F.conv2d(x_gray, self.sobel_x, padding=1)
                gy = F.conv2d(x_gray, self.sobel_y, padding=1)
                edge = torch.sqrt(gx * gx + gy * gy + 1e-6)  # (B,1,H,W)
                # 旋转到与当前分支一致
                edge_k = torch.rot90(edge, k=k, dims=(2, 3))
                # 与分支输出对齐的空间尺寸（若 stride>1 等情况）
                if edge_k.shape[-2:] != yk.shape[-2:]:
                    edge_k = F.interpolate(edge_k, size=yk.shape[-2:], mode="bilinear", align_corners=False)
                # 投影到输出通道并生成 [0,1] 的调制掩码
                mask = self.edge_proj(edge_k)       # (B,C_out,H_out,W_out)
                mask = self.edge_bn(mask)
                mask = self.edge_act(mask)          # (B,C_out,H_out,W_out)
                # 高频增强（逐像素）
                yk = yk + self.gamma * mask * yk

            # 7) 按方向注意力进行加权融合
            wk_attn = dir_weights[:, k:k+1, :, :]  # (B,1,1,1)
            # 通过广播，将 (B,1,1,1) 作用到 (B,C_out,H_out,W_out)
            yk = yk * wk_attn

            y_sum = yk if (y_sum is None) else (y_sum + yk)

        return y_sum


# ----------------------------- 自测脚本 -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入张量：B, C, H, W
    x = torch.randn(2, 32, 256, 256, device=device)

    # 构建 SPARC-Conv
    net = SPARCConv(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_edge_boost=True,  # 可切换 False 关闭高频强调，便于对比
    ).to(device)

    # 前向推理；不指定 output_shape 则默认保持空间尺寸
    y = net(x, output_shape=(256, 256))

    # 打印信息
    print(net)
    print("\n[模块命名] SPARC-Conv: Self-Padded Adaptive Rotation-Consistent Convolution")
    print("[创新点] 方向注意力 + 双模式自适应填充 + 高频边缘强调（Sobel引导）\n")
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    # 期望：Output 与 Input 的 (B, C, H, W) 一致
