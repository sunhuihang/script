import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class MiLKConvAttn(nn.Module):
    """
    MiLK-ConvAttn: Mixture-of-Large-Kernels Convolutional Attention
    - Long-range path: 从一个小型“大核字典”按样本×通道路由出 depthwise 大核，建模长程依赖
    - Local path: 动态 3x3 depthwise 卷积，捕获局部细节（实例敏感）
    - Bi-path gating: 两分支通道级软门控（softmax），自适应融合
    - Optional external path: 兼容外部共享大核 lk_filter（full conv），用于复现实验/对比
    """
    def __init__(
        self,
        pdim: int,
        proj_dim_in: Optional[int] = None,
        k: int = 13,             # 大核尺寸
        num_bases: int = 8,      # 大核字典规模
        sk_size: int = 3         # 动态小核尺寸
    ):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.k = k
        self.num_bases = num_bases
        self.sk_size = sk_size

        hidden = max(16, self.proj_dim_in // 2)

        # ---- (1) 大核字典：共享、可学习、通道无关（depthwise 的核）----
        # 形状 [num_bases, 1, k, k]
        self.lk_bases = nn.Parameter(torch.randn(num_bases, 1, k, k) * 0.01)

        # ---- (2) 路由器：输出 [B, pdim, num_bases] 的权重，对字典做加权混合 ----
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * num_bases, 1)
        )

        # ---- (3) 动态 3×3 depthwise 小核（局部分支）----
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * self.sk_size * self.sk_size, 1)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

        # ---- (4) 双路径软门控（每通道 2 个门控值，softmax）----
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * 2, 1)
        )

        # ---- (5) 融合投影（1×1），随后与残差相加 ----
        self.fuse = nn.Conv2d(pdim, pdim, 1)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, C, H, W]
        lk_filter (optional): [pdim, pdim, k, k] 的外部共享大核（full conv），作为额外路径
        """
        B, C, H, W = x.shape
        assert C >= self.pdim, "Input channels must be >= pdim"

        # 拆分注意力通道与旁路通道
        x1, x2 = x[:, :self.pdim], x[:, self.pdim:]

        # ---------- 长程分支：由字典路由得到每样本×每通道的大核 ----------
        # 路由权重 [B, pdim, num_bases]
        rw = self.router(x[:, :self.proj_dim_in]).view(B, self.pdim, self.num_bases)
        rw = torch.softmax(rw, dim=-1)

        # 合成 depthwise 大核 [B, pdim, 1, k, k]：对字典做加权求和
        # self.lk_bases: [num_bases, 1, k, k]
        composed_k = (rw[..., None, None, None] * self.lk_bases[None, None, ...]).sum(dim=2)

        # 使用合成大核做 depthwise 卷积（groups = B * pdim）
        x1_reshaped = rearrange(x1, 'b c h w -> 1 (b c) h w')
        composed_k_groups = rearrange(composed_k, 'b c o k1 k2 -> (b c) o k1 k2')
        out_lk = F.conv2d(x1_reshaped, composed_k_groups, padding=self.k // 2, groups=B * self.pdim)
        out_lk = rearrange(out_lk, '1 (b c) h w -> b c h w', b=B, c=self.pdim)

        # ---------- 局部分支：动态 3×3 depthwise ----------
        dyn_k = self.dwc_proj(x[:, :self.proj_dim_in]).view(B, self.pdim, 1, self.sk_size, self.sk_size)
        dyn_k = rearrange(dyn_k, 'b c o k1 k2 -> (b c) o k1 k2')
        out_dyn = F.conv2d(x1_reshaped, dyn_k, padding=self.sk_size // 2, groups=B * self.pdim)
        out_dyn = rearrange(out_dyn, '1 (b c) h w -> b c h w', b=B, c=self.pdim)

        # ---------- 可选外部 full-conv 路径（与原实现兼容） ----------
        if lk_filter is not None:
            out_ext = F.conv2d(x1, lk_filter, padding=lk_filter.shape[-1] // 2)
        else:
            out_ext = 0.0  # 标量 0，直接广播

        # ---------- 双路径软门控融合（外部路径并入长程分支） ----------
        g = self.gate(x[:, :self.proj_dim_in]).view(B, 2, self.pdim, 1, 1)
        g = torch.softmax(g, dim=1)
        g_lk, g_dyn = g[:, 0], g[:, 1]

        fused = g_lk * (out_lk + (out_ext if isinstance(out_ext, torch.Tensor) else 0.0)) + g_dyn * out_dyn

        # ---------- 1×1 融合投影 & 残差 ----------
        y1 = self.fuse(fused) + x1

        # 复原通道
        y = torch.cat([y1, x2], dim=1)
        return y

    def extra_repr(self):
        return f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}, k={self.k}, num_bases={self.num_bases}, sk_size={self.sk_size}'


if __name__ == "__main__":
    torch.manual_seed(42)

    # 配置
    batch_size = 1
    channels = 64     # 总通道
    height = 128
    width = 128
    pdim = 32         # 参与注意力/大核的前半通道

    # 输入张量 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)

    # 可选：外部共享大核（full conv） [outC, inC, k, k]，保持与原代码接口兼容
    lk_filter = torch.randn(pdim, pdim, 13, 13)

    # 实例化模型（proj_dim_in 可设为 <= C，用于生成门控/路由的上下文输入维度）
    model = MiLKConvAttn(pdim=pdim, proj_dim_in=48, k=13, num_bases=6, sk_size=3)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    lk_filter = lk_filter.to(device)
    model = model.to(device)

    # 前向传播
    with torch.no_grad():
        output = model(x, lk_filter)  # lk_filter 可传 None 做纯 MiLK-ConvAttn

    # 打印信息
    print(model)
    print("\n=== Shapes ===")
    print("Input :", tuple(x.shape))
    print("\n微信公众号|Bilibili CV缝合救星")
    print("Output:", tuple(output.shape))
