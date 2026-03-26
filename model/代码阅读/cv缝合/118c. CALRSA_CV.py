import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContentAdaptivePool(nn.Module):
    def __init__(self, mode_size_hw: tuple[int, int]):
        super().__init__()
        self.size = mode_size_hw
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, self.size)
        mx = F.adaptive_max_pool2d(x, self.size)
        hint = torch.cat([avg.mean(1, keepdim=True), mx.mean(1, keepdim=True)], dim=1)
        alpha = self.gate(hint)
        out = alpha * mx + (1 - alpha) * avg
        return out


class HighResFeedbackGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, hi_res_x: torch.Tensor, up_feat: torch.Tensor) -> torch.Tensor:
        gate = self.spatial_gate(hi_res_x)
        return up_feat * (1.0 + gate)


class CALRSA_BCHW(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        q_pooled_size: int = 16,
        kv_pooled_sizes=(11, 8, 6, 4),
        topk_ratio: float = 0.75,
        use_hr_feedback: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.kv_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_pool_size = q_pooled_size
        self.q_pool_adapt = None

        self.kv_sizes = list(kv_pooled_sizes)
        self.eps = 1e-6

        self.kv_dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for _ in self.kv_sizes
        ])

        self.topk_ratio = float(topk_ratio)
        self.token_scorer = nn.Linear(dim, 1, bias=False)

        self.use_hr_feedback = use_hr_feedback
        if use_hr_feedback:
            self.hr_gate = HighResFeedbackGate(dim)

        self.pre_dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.post_dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _keep_aspect_size(H: int, W: int, target: int) -> tuple[int, int]:
        if W >= H:
            h = target
            w = max(1, round(W * (target / (H + 1e-6))))
        else:
            w = target
            h = max(1, round(H * (target / (W + 1e-6))))
        return h, w

    def _content_adaptive_q(self, feat: torch.Tensor):
        B, C, H, W = feat.shape
        feat_enh = self.pre_dwconv(feat)
        qh, qw = self._keep_aspect_size(H, W, self.q_pool_size)
        if self.q_pool_adapt is None or self.q_pool_adapt.size != (qh, qw):
            self.q_pool_adapt = ContentAdaptivePool((qh, qw))
        q_low = self.q_pool_adapt(feat_enh)
        q_tokens = q_low.flatten(2).transpose(1, 2)
        q = self.q_proj(q_tokens)
        q = q.view(B, qh * qw, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return q, (qh, qw)

    def _pyramid_kv(self, feat: torch.Tensor):
        B, C, H, W = feat.shape
        kv_tokens_list = []
        for ps, dw in zip(self.kv_sizes, self.kv_dwconvs):
            kh, kw = self._keep_aspect_size(H, W, ps)
            pooled = F.adaptive_avg_pool2d(feat, (kh, kw))
            pooled = dw(pooled)
            kv_tokens_list.append(pooled.flatten(2))
        kv_feat = torch.cat(kv_tokens_list, dim=2).transpose(1, 2)
        kv_feat = self.kv_norm(kv_feat)
        Lk = kv_feat.shape[1]
        keep_k = max(1, int(np.ceil(self.topk_ratio * Lk)))
        with torch.no_grad():
            scores = self.token_scorer(kv_feat)
            idx = torch.topk(scores.squeeze(-1), k=keep_k, dim=1, largest=True, sorted=False).indices
        batch_indices = torch.arange(B, device=kv_feat.device).unsqueeze(-1).expand(B, keep_k)
        kv_kept = kv_feat[batch_indices, idx]
        kv = self.kv_proj(kv_kept).view(B, keep_k, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]
        return k, v, keep_k

    def forward(self, x: torch.Tensor):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        q, (qh, qw) = self._content_adaptive_q(x)
        k, v, _ = self._pyramid_kv(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, qh * qw, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, qh, qw)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        if self.use_hr_feedback:
            out = self.hr_gate(x, out)
        out = self.post_dwconv(out)
        return out


if __name__ == "__main__":
    B, C, H, W = 2, 64, 32, 48
    x = torch.randn(B, C, H, W)
    attn = CALRSA_BCHW(
        dim=C,
        num_heads=4,
        qkv_bias=True,
        q_pooled_size=16,
        kv_pooled_sizes=(11, 8, 6, 4),
        topk_ratio=0.7,
        use_hr_feedback=True
    )
    y = attn(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
