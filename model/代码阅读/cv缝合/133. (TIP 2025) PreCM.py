# -*- coding: utf-8 -*-
# PreCM: Padding-based Rotation Equivariant Convolution Mode (minimal runnable)
# 运行环境：Python 3.8+，PyTorch 1.10+

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def _rotate_padding_vector(p: Tuple[int, int, int, int], k: int) -> Tuple[int, int, int, int]:
    """
    输入/返回格式均为 (pa, pl, pb, pr) = (上、左、下、右)。
    每顺时针旋转 90°，根据论文置换： (a, l, b, r) -> (r, a, l, b)。
    连续应用 k 次。
    """
    pa, pl, pb, pr = p
    for _ in range(k % 4):
        pa, pl, pb, pr = pr, pa, pl, pb
    return pa, pl, pb, pr

class PreCM(nn.Module):
    """
    PreCM: 基于填充的旋转等变卷积模块
    - 4 个旋转分支（0/90/180/270）：
        * 对输入、卷积核、padding 同步旋转
        * 卷积输出逆旋转对齐
        * 等权融合（sum 或 mean）
    - 输出的 batch 与通道数 与输入一致
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 fuse: str = "sum"  # "sum" 或 "mean"
                 ):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0, "groups 必须整除通道数"
        assert fuse in ("sum", "mean"), "fuse 只能为 'sum' 或 'mean'"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.groups = int(groups)
        self.dilation = int(dilation)
        self.fuse = fuse

        # 卷积核参数（σ0 基准方向）
        w = torch.empty(out_channels, in_channels // groups, self.kernel_size, self.kernel_size)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight0 = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    @staticmethod
    def _compute_padding(h_in: int, w_in: int,
                         k: int, s: int, d: int,
                         h_out: int, w_out: int) -> Tuple[int, int, int, int]:
        """
        依据论文式(22)计算 σ0 的 padding：
        返回 (pa, pl, pb, pr) = (上、左、下、右)
        """
        pab = (h_out - 1) * s + d * (k - 1) + 1 - h_in
        prl = (w_out - 1) * s + d * (k - 1) + 1 - w_in
        pb = pab // 2
        pl = prl // 2
        pa = pab - pb
        pr = prl - pl
        # 负 padding 置 0（如需严格裁剪，可在卷积后裁剪）
        pa = int(max(pa, 0)); pb = int(max(pb, 0))
        pl = int(max(pl, 0)); pr = int(max(pr, 0))
        return pa, pl, pb, pr

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        x: (B, C, H, W)
        output_shape: (H_out, W_out)。不传则默认保持 H_out=H, W_out=W。
        """
        B, C, H, W = x.shape
        if output_shape is None:
            h_out, w_out = H, W
        else:
            h_out, w_out = int(output_shape[0]), int(output_shape[1])

        # σ0 基础 padding
        pa0, pl0, pb0, pr0 = self._compute_padding(
            H, W, self.kernel_size, self.stride, self.dilation, h_out, w_out
        )

        y_list = []
        for k in range(4):  # 0°,90°,180°,270°
            # 1) 旋转输入
            xk = torch.rot90(x, k=k, dims=(2, 3))
            # 2) 旋转 padding，并按 PyTorch F.pad 顺序 (left, right, top, bottom) 应用
            pak, plk, pbk, prk = _rotate_padding_vector((pa0, pl0, pb0, pr0), k)
            pad_tuple = (plk, prk, pak, pbk)
            if any(v > 0 for v in pad_tuple):
                xk = F.pad(xk, pad_tuple, mode="constant", value=0.0)
            # 3) 旋转卷积核
            wk = torch.rot90(self.weight0, k=k, dims=(2, 3))
            # 4) 卷积
            yk = F.conv2d(
                xk, weight=wk, bias=self.bias,
                stride=self.stride, dilation=self.dilation, groups=self.groups
            )
            # 5) 逆旋转回参考方向
            yk = torch.rot90(yk, k=(4 - k) % 4, dims=(2, 3))
            y_list.append(yk)

        y = torch.stack(y_list, dim=0).sum(dim=0)
        if self.fuse == "mean":
            y = y / 4.0
        return y


# ----------------- 简单自测 -----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入张量：B, C, H, W
    x = torch.randn(1, 32, 256, 256, device=device)

    # 构建 PreCM
    net = PreCM(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        fuse="sum"  # 或 "mean"
    ).to(device)

    # 前向
    y = net(x, output_shape=(256, 256))

    print(net)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, PreCM 复现\n")
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    # 期望输出：Output 与 Input 的 (B, C, H, W) 一致
