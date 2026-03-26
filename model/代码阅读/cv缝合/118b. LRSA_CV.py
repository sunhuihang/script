import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LRSA_BCHW(nn.Module):
    """
    LRSA 的 BCHW 版：
    输入:  x  [B, C, H, W]
    输出:  y  [B, C, H, W]
    其它行为尽量与题主原始实现保持一致：
      - Q: 固定低分辨率池化 (可保持长宽比)
      - K/V: 多尺度自适应平均池化 + 深度卷积增强
      - 注意力在低分辨率空间计算，必要时上采样回 HxW
    """
    def __init__(
        self,
        dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        pooled_sizes=(11, 8, 6, 4),
        q_pooled_size=1,
        q_conv=False
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = list(pooled_sizes)
        self.eps = 1e-3

        self.norm = nn.LayerNorm(dim)

        self.q_pooled_size = int(q_pooled_size)

        # 可选：在 Q 分支前做深度卷积增强（仅当 q_pooled_size > 1 才有意义）
        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False)
            self.q_bn = nn.BatchNorm2d(dim)
            self.q_act = nn.ReLU(inplace=True)
            self.q_ln = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_bn = None
            self.q_act = None
            self.q_ln = None

    @staticmethod
    def _keep_aspect_size(H: int, W: int, target: int, eps: float = 1e-6):
        """按照原实现的思路，保持长宽比地产生目标池化尺寸。"""
        if target <= 1:  # 特判：等同于不池化（返回原尺寸）
            return H, W
        if W >= H:
            h = target
            w = max(1, round(W * (target / (H + eps))))
        else:
            w = target
            h = max(1, round(H * (target / (W + eps))))
        return int(h), int(w)

    def forward(self, x: torch.Tensor, d_convs=None):
        """
        x:       [B, C, H, W]
        d_convs: nn.ModuleList，与 pooled_sizes 一一对应的 DWConv (可选)。若为 None，将内部创建 Identity。
        return:  [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.dim, f"Input channels {C} must match dim {self.dim}"

        # ---------- Q 分支：固定低分辨率池化 ----------
        qH, qW = self._keep_aspect_size(H, W, self.q_pooled_size, self.eps)
        if self.q_pooled_size > 1:
            q_feat = F.adaptive_avg_pool2d(x, (qH, qW))  # [B, C, qH, qW]
            if self.q_conv is not None:
                q_feat = self.q_act(self.q_bn(self.q_conv(q_feat)))
            q_tokens = q_feat.flatten(2).transpose(1, 2)  # [B, qH*qW, C]
            q_tokens = self.q(q_tokens)                   # 线性映射
        else:
            # 不池化：直接使用原分辨率
            q_tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            if self.q_conv is not None:
                # 如果硬要启用 q_conv，这里也做一下（与原逻辑兼容）
                x_enh = self.q_act(self.q_bn(self.q_conv(x)))
                q_tokens = x_enh.flatten(2).transpose(1, 2)
                q_tokens = self.q(q_tokens)
            else:
                q_tokens = self.q(q_tokens)

        # multi-head 形状：[B, heads, Lq, head_dim]
        q = q_tokens.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # ---------- K/V 分支：Pyramid pooling + (可选)DWConv ----------
        if d_convs is None:
            d_convs = [nn.Identity() for _ in self.pooled_sizes]
        assert len(d_convs) == len(self.pooled_sizes), "d_convs must match pooled_sizes length"

        kv_pools = []
        for ps, l in zip(self.pooled_sizes, d_convs):
            kH, kW = self._keep_aspect_size(H, W, int(ps), self.eps)
            pool = F.adaptive_avg_pool2d(x, (kH, kW))   # [B, C, kH, kW]
            pool = l(pool)                               # DWConv or Identity
            kv_pools.append(pool.flatten(2))             # [B, C, kH*kW]

        kv_cat = torch.cat(kv_pools, dim=2).transpose(1, 2)  # [B, Lkv, C]
        kv_cat = self.norm(kv_cat)                           # LN on last dim

        kv = self.kv(kv_cat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, heads, Lkv, head_dim]

        # ---------- Self-Attention (低分辨率) ----------
        attn = (q @ k.transpose(-2, -1)) * self.scale        # [B, heads, Lq, Lkv]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = attn @ v                                        # [B, heads, Lq, head_dim]
        y = y.transpose(1, 2).reshape(B, -1, C)             # [B, Lq, C]
        y = self.proj(y)
        y = self.proj_drop(y)

        # ---------- 回到 BCHW ----------
        if self.q_pooled_size > 1:
            y = y.transpose(1, 2).reshape(B, C, qH, qW)     # [B, C, qH, qW]
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        else:
            y = y.transpose(1, 2).reshape(B, C, H, W)       # [B, C, H, W]

        return y


# ======================== 最小可运行示例 ========================
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 1          # batch size
    C = 64         # embedding dim / channels
    H, W = 32, 32  # height and width

    # 输入：BCHW
    x = torch.randn(B, C, H, W)

    # 与 pooled_sizes 一一对应的深度卷积；若不传可为 None
    pooled_sizes = [11, 8, 6, 4]
    d_convs = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        ) for _ in pooled_sizes
    ])

    # 实例化 BCHW 版 LRSA
    attn = LRSA_BCHW(
        dim=C,
        num_heads=4,
        qkv_bias=True,
        pooled_sizes=pooled_sizes,
        q_pooled_size=1,   # =1 表示不对 Q 降采样；>1 表示将 Q 固定池化到约 q_pooled_size×q_pooled_size
        q_conv=False       # 可设 True 测试 Q 分支的DWConv增强
    )

    out = attn(x, d_convs=d_convs)

    print(attn)
    print("\n微信公众号:CV缝合救星\n")
    print(f"Input shape : {x.shape}")   # [B, C, H, W]
    print(f"Output shape: {out.shape}") # [B, C, H, W]
