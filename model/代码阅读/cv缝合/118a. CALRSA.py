import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContentAdaptivePool(nn.Module):
    """
    内容自适应池化：AvgPool + MaxPool 双分支，Sigmoid 门控自适应融合。
    目标：比单一 Avg 更稳健，保留显著区域与背景抑制能力。
    """
    def __init__(self, mode_size_hw: tuple[int, int]):
        super().__init__()
        self.size = mode_size_hw  # (h, w)
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),  # 在通道维做门控（用avg/max的2通道提示）
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg = F.adaptive_avg_pool2d(x, self.size)
        mx  = F.adaptive_max_pool2d(x, self.size)
        # 计算门控（不依赖通道，使用2张提示图在空间上产生门控）
        hint = torch.cat([avg.mean(1, keepdim=True), mx.mean(1, keepdim=True)], dim=1)  # [B,2,h,w]
        alpha = self.gate(hint)  # [B,1,h,w]
        out = alpha * mx + (1 - alpha) * avg
        return out


class HighResFeedbackGate(nn.Module):
    """
    高分辨率反馈调制：用高分辨率特征生成空间门控，调制上采样后的注意力输出。
    轻量、稳定、对边界敏感。
    """
    def __init__(self, channels):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, hi_res_x: torch.Tensor, up_feat: torch.Tensor) -> torch.Tensor:
        # hi_res_x: [B, C, H, W]  (来自输入的高分辨率特征重新reshape)
        # up_feat : [B, C, H, W]  (上采样后的注意力结果)
        gate = self.spatial_gate(hi_res_x)  # [B,1,H,W]
        return up_feat * (1.0 + gate)      # 残差门控：避免过度抑制


class CALRSA(nn.Module):
    """
    CALRSA: Content-Adaptive Low-Resolution Self-Attention
    - 内容自适应 Q 池化 (avg/max gate)
    - K/V 金字塔池化 + DWConv
    - 动态令牌选择 (Top-k)
    - 高分辨率反馈调制 (HR Feedback)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        q_pooled_size: int = 16,            # Q 的目标 pooled size（边长）
        kv_pooled_sizes=(11, 8, 6, 4),      # K/V 的多尺度 pooled size（边长）
        topk_ratio: float = 0.75,           # 动态令牌选择比例（0~1]
        use_hr_feedback: bool = True,       # 是否启用高分辨率反馈
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q/KV 映射
        self.q_proj  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 归一化与投影
        self.kv_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 自适应 Q 池化（保持长宽比）
        self.q_pool_size = q_pooled_size
        self.q_pool_adapt = None  # 实例化时根据输入HW算比例

        # K/V 金字塔池化配置
        self.kv_sizes = list(kv_pooled_sizes)
        self.eps = 1e-6

        # 深度可分离卷积用于局部增强（每个尺度一个）
        self.kv_dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for _ in self.kv_sizes
        ])

        # 动态令牌选择
        self.topk_ratio = float(topk_ratio)
        assert 0.0 < self.topk_ratio <= 1.0, "topk_ratio must be (0, 1]."
        self.token_scorer = nn.Linear(dim, 1, bias=False)

        # 高分辨率反馈门控
        self.use_hr_feedback = use_hr_feedback
        if use_hr_feedback:
            self.hr_gate = HighResFeedbackGate(dim)

        # 局部细节增强（放在注意力前后）
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

    def _content_adaptive_q(self, x: torch.Tensor, H: int, W: int):
        """
        生成内容自适应的 Q 令牌
        x: [B, N, C]
        return: q_tokens [B, Lq, C], (h, w)
        """
        B, N, C = x.shape
        feat = x.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]

        # 预先用 DWConv 提升局部细节
        feat_enh = self.pre_dwconv(feat)

        # 自适应池化尺寸（保持长宽比）
        qh, qw = self._keep_aspect_size(H, W, self.q_pool_size)
        if self.q_pool_adapt is None or self.q_pool_adapt.size != (qh, qw):
            self.q_pool_adapt = ContentAdaptivePool((qh, qw))

        q_low = self.q_pool_adapt(feat_enh)  # [B,C,qh,qw]
        q_tokens = q_low.flatten(2).transpose(1, 2)   # [B, qh*qw, C]
        q = self.q_proj(q_tokens)                     # 线性映射
        # reshape to multi-head
        q = q.view(B, qh * qw, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # [B,h,Lq,d]
        return q, (qh, qw)

    def _pyramid_kv(self, x: torch.Tensor, H: int, W: int):
        """
        金字塔池化生成 K/V 令牌，并进行动态令牌选择
        x: [B, N, C]
        return: k [B,h,Lk,d], v [B,h,Lk,d], (list of (sh,sw)) for analysis
        """
        B, N, C = x.shape
        feat = x.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]

        kv_tokens_list = []
        shapes = []
        for ps, dw in zip(self.kv_sizes, self.kv_dwconvs):
            kh, kw = self._keep_aspect_size(H, W, ps)
            pooled = F.adaptive_avg_pool2d(feat, (kh, kw))
            pooled = dw(pooled)  # 局部增强
            kv_tokens_list.append(pooled.flatten(2))  # [B,C,kh*kw]
            shapes.append((kh, kw))

        kv_feat = torch.cat(kv_tokens_list, dim=2).transpose(1, 2)  # [B, Lk, C]
        kv_feat = self.kv_norm(kv_feat)

        # 动态令牌选择：根据 kv_feat 的显著性打分，选择 Top-k
        Lk = kv_feat.shape[1]
        keep_k = max(1, int(np.ceil(self.topk_ratio * Lk)))
        with torch.no_grad():
            scores = self.token_scorer(kv_feat)  # [B,Lk,1]
            idx = torch.topk(scores.squeeze(-1), k=keep_k, dim=1, largest=True, sorted=False).indices  # [B,keep_k]
        # 为了可微性，索引仍可传播到被选中的元素（对未选中的停止梯度）
        batch_indices = torch.arange(B, device=kv_feat.device).unsqueeze(-1).expand(B, keep_k)
        kv_kept = kv_feat[batch_indices, idx]  # [B,keep_k,C]

        kv = self.kv_proj(kv_kept).view(B, keep_k, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]  # [B,h,keep_k,d], [B,h,keep_k,d]
        return k, v, keep_k

    def forward(self, x: torch.Tensor, H: int, W: int, d_convs=None):
        """
        x: [B, N, C]  (N=H*W)
        return: [B, N, C]
        """
        B, N, C = x.shape
        # 1) 内容自适应 Q
        q, (qh, qw) = self._content_adaptive_q(x, H, W)  # q: [B,h,Lq,d]

        # 2) 金字塔 K/V + 动态令牌选择
        k, v, keep_k = self._pyramid_kv(x, H, W)  # [B,h,Lk',d]

        # 3) 低分辨率注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,h,Lq,Lk']
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B,h,Lq,d]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, qh * qw, C)  # [B,Lq,C]
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # 4) 上采样回高分辨率，并做高分辨率反馈调制
        out = out.transpose(1, 2).reshape(B, C, qh, qw)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # 高分辨率反馈门控
        if self.use_hr_feedback:
            hi = x.transpose(1, 2).reshape(B, C, H, W)
            out = self.hr_gate(hi, out)

        # 5) 末端再做一次轻量局部增强
        out = self.post_dwconv(out)

        # 6) 展平成 [B,N,C]
        out = out.flatten(2).transpose(1, 2)
        return out


# --------- 测试用最小可运行示例 ---------
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    C = 64
    H, W = 32, 48
    N = H * W

    x = torch.randn(B, N, C)

    # 实例化 CALRSA（可直接替换原 LRSA）
    attn = CALRSA(
        dim=C,
        num_heads=4,
        qkv_bias=True,
        q_pooled_size=16,         # Q 的目标 pooled 边长
        kv_pooled_sizes=(11,8,6,4),
        topk_ratio=0.7,           # 仅使用 70% 的 K/V 令牌参与注意力
        use_hr_feedback=True,
        attn_drop=0.0,
        proj_drop=0.0
    )

    out = attn(x, H, W, d_convs=None)

    print(attn)
    print("\n微信公众号:CV缝合救星\n")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")  # [B, N, C]
