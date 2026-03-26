import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 基础模块
# -----------------------------
class ChannelAttention(nn.Module):
    """轻量通道注意力：Avg/Max + 1x1 -> ReLU -> 1x1 -> Sigmoid"""
    def __init__(self, inp: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, inp // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(inp, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        maxv = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        w = self.sigmoid(avg + maxv)
        return x * w


class _MultiScaleStrip1D(nn.Module):
    """
    多尺度条带 1D 深度可分离卷积（沿序列维度），分支 dilation=[1,2,3]。
    输入: (B, C, L) ; groups=C 做 depthwise; 输出同形状。
    """
    def __init__(self, channels: int, kernel_size: int = 7, dilations=(1, 2, 3)):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.branches = nn.ModuleList()
        for d in dilations:
            pad = d * (kernel_size - 1) // 2
            self.branches.append(
                nn.Conv1d(channels, channels, kernel_size=kernel_size,
                          padding=pad, dilation=d, groups=channels, bias=False)
            )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sum 多分支
        y = None
        for conv in self.branches:
            out = conv(x)
            y = out if y is None else (y + out)
        y = self.bn(y)
        y = self.act(y)
        return y  # (B, C, L), 值域[0,1]


class _DirectionalWeight(nn.Module):
    """
    单方向（高 或 宽）权重生成：
    - 先在“正交维”做 max/avg 池化（2 通道） -> 1x1 conv 压到 1 通道
    - view 成 (B, C, L) 后用多尺度条带 1D 卷积生成逐通道序列权重
    - 输出回到 (B, 1, C, L) 以便广播到原格式
    约定输入 x 的形状为 (B, L_ortho, C, L)：
       若做“高度方向”的权重，则 L 为 H，L_ortho 为 W；
       若做“宽度方向”的权重，则 L 为 W，L_ortho 为 H。
    """
    def __init__(self, channels: int, kernel_size: int = 7, dilations=(1, 2, 3)):
        super().__init__()
        self.squeeze = nn.Conv2d(2, 1, kernel_size=1, bias=True)  # 压 2->1
        self.strip = _MultiScaleStrip1D(channels, kernel_size=kernel_size, dilations=dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L_ortho, C, L)
        b, l_ortho, c, l = x.size()
        # 沿正交维（dim=1）做 max/avg，得到 (B,1,C,L)
        pooled_max = torch.max(x, dim=1, keepdim=True)[0]
        pooled_avg = torch.mean(x, dim=1, keepdim=True)
        pooled = torch.cat([pooled_max, pooled_avg], dim=1)   # (B,2,C,L)
        s = self.squeeze(pooled)                               # (B,1,C,L)

        # 转成 (B,C,L)，做多尺度 strip-1D 权重
        s = s.view(b, c, l)                                    # (B,C,L)
        w = self.strip(s)                                      # (B,C,L) in [0,1]
        w = w.view(b, 1, c, l)                                 # (B,1,C,L)
        return w


# -----------------------------
# 魔改版 IIA：双向多尺度 + 跨向量耦合 + 通道注意 + 可学习残差缩放
# -----------------------------
class IIA_X(nn.Module):
    """
    Innovation IIA (IIA_X):
    - 双向（H/W）多尺度条带注意力
    - Cross-Orientation Gating: 自适应混合 λ_h, λ_w
    - ChannelAttention 协同重标定
    - 可学习残差缩放 α, β
    输入/输出: (B, C, H, W)
    """
    def __init__(self, channels: int, ksize: int = 7, dilations=(1, 2, 3), ca_ratio: int = 16):
        super().__init__()
        self.h_weight = _DirectionalWeight(channels, kernel_size=ksize, dilations=dilations)
        self.w_weight = _DirectionalWeight(channels, kernel_size=ksize, dilations=dilations)

        # 跨方向耦合门：将两方向的全局描述拼接 -> 2通道 -> 2通道，再做 Softmax 得到 [λ_h, λ_w]
        self.gate_fc = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=channels, bias=False),  # depthwise
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.mix_linear = nn.Linear(2, 2, bias=True)  # 对每个通道的 [g_h, g_w] 计算混合系数

        self.ca = ChannelAttention(channels, ratio=ca_ratio)

        # 可学习残差缩放
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 方向注意注入强度
        self.beta = nn.Parameter(torch.tensor(1.0))   # 通道注意注入强度

    def _orientation_mix(self, wh: torch.Tensor, ww: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        根据两方向的全局序列描述自适应出 λ_h / λ_w：
        - 先对 (B, C, L) 做 depthwise 1x1 + BN + ReLU
        - 沿 L 做自适应平均得到 (B, C) 的描述
        - 拼 (B, C, 2)，过 Linear(2->2)，对最后维 softmax -> 两方向权重
        返回: (lambda_h, lambda_w), 形状均为 (B, C, 1) 以便广播
        """
        # wh, ww: (B, C, L)
        gh = self.gate_fc(wh)              # (B,C,L)
        gw = self.gate_fc(ww)              # (B,C,L)
        gh = gh.mean(dim=-1)               # (B,C)
        gw = gw.mean(dim=-1)               # (B,C)
        g = torch.stack([gh, gw], dim=-1)  # (B,C,2)
        logits = self.mix_linear(g)        # (B,C,2)
        lambdas = F.softmax(logits, dim=-1)
        lambda_h = lambdas[..., 0].unsqueeze(-1)  # (B,C,1)
        lambda_w = lambdas[..., 1].unsqueeze(-1)  # (B,C,1)
        return lambda_h, lambda_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.size()

        # ---- 高度方向 ----
        # 组织为 (B, W, C, H)，使“序列维=L=H”，正交维为 W
        x_h_in = x.permute(0, 3, 1, 2).contiguous()
        w_h = self.h_weight(x_h_in)                    # (B,1,C,H)
        # 注意力施加并还原
        x_h_out = (x_h_in * w_h).permute(0, 2, 3, 1).contiguous()  # -> (B,C,H,W)

        # ---- 宽度方向 ----
        # 组织为 (B, H, C, W)，使“序列维=L=W”，正交维为 H
        x_w_in = x.permute(0, 2, 1, 3).contiguous()
        w_w = self.w_weight(x_w_in)                    # (B,1,C,W)
        x_w_out = (x_w_in * w_w).permute(0, 2, 1, 3).contiguous()  # -> (B,C,H,W)

        # ---- 跨方向耦合门（从权重里提序列做自适应混合）----
        wh_seq = w_h.view(b, c, h)   # (B,C,H)
        ww_seq = w_w.view(b, c, w)   # (B,C,W)
        lambda_h, lambda_w = self._orientation_mix(wh_seq, ww_seq)  # (B,C,1),(B,C,1)

        # 广播到 (B,C,H,W)
        x_h_mix = x_h_out * lambda_h.unsqueeze(-1)     # (B,C,H,W)
        x_w_mix = x_w_out * lambda_w.unsqueeze(-1)     # (B,C,H,W)

        # ---- 通道注意力融合 ----
        x_ca = self.ca(x)

        # ---- 残差融合 ----
        out = x + self.alpha * (x_h_mix + x_w_mix) + self.beta * x_ca
        return out


# -----------------------------
# 自测
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W, device=device)

    model = IIA_X(channels=C, ksize=7, dilations=(1, 2, 3), ca_ratio=16).to(device)
    y = model(x)

    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量形状:", tuple(x.shape))
    print("输出张量形状:", tuple(y.shape))
    # 简单数值 sanity check
    print("输出均值/方差:", y.mean().item(), y.std().item())
