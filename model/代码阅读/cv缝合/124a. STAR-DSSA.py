import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from typing import Optional, Tuple

# =============== 工具函数 ===============

def _grid2seq(x: Tensor, region_size: Tuple[int, int], num_heads: int):
    """
    功能：把输入图像划分成若干区域，并转换成序列形式，方便做区域级注意力
    输入:
        x: 输入张量，形状 (B, C, H, W)
        region_size: 每个区域的高和宽
        num_heads: 注意力头数
    输出:
        out: 转换后的序列张量，形状 (B, nH, nRegion, reg_size^2, head_dim)
        region_h, region_w: 每行/每列的区域数
    """
    B, C, H, W = x.size()
    rh, rw = region_size
    region_h, region_w = H // rh, W // rw
    # (B, C, H, W) -> (B, nH, C//nH, region_h, rh, region_w, rw)
    x = x.view(B, num_heads, C // num_heads, region_h, rh, region_w, rw)
    # 维度重排 -> (B, nH, region_h, region_w, rh, rw, head_dim)
    x = torch.einsum("bmdhpwq->bmhwpqd", x)
    # 拉直 -> (B, nH, nRegion, reg_size^2, head_dim)
    x = x.flatten(2, 3).flatten(-3, -2)
    return x, region_h, region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int, int]):
    """
    功能：把序列形式的特征恢复成原始网格图像形式
    输入:
        x: 输入张量 (B, nH, nRegion, reg_size^2, head_dim)
    输出:
        x: 恢复后的图像特征 (B, C, H, W)
    """
    B, nH, nRegion, reg_sz_sq, head_dim = x.size()
    rh, rw = region_size
    # 恢复形状
    x = x.view(B, nH, region_h, region_w, rh, rw, head_dim)
    x = torch.einsum("bmhwpqd->bmdhpwq", x)
    C = nH * head_dim
    H = region_h * rh
    W = region_w * rw
    x = x.reshape(B, C, H, W)
    return x


def _build_adjacency_bias(region_h: int, region_w: int, radius: int = 1, device=None):
    """
    功能：构造区域邻接矩阵，用于在区域路由中引入空间先验
    输入:
        region_h, region_w: 区域划分的行列数
        radius: 邻域半径（切比雪夫距离）
    输出:
        A: 邻接矩阵 (R, R)，R=region_h*region_w
    """
    R = region_h * region_w
    coords = torch.stack(torch.meshgrid(
        torch.arange(region_h), torch.arange(region_w), indexing='ij'
    ), dim=-1).reshape(-1, 2)  # (R, 2)
    diff = coords[:, None, :] - coords[None, :, :]  # (R, R, 2)
    dist = diff.abs().max(dim=-1).values  # 切比雪夫距离
    A = (dist <= radius).float()
    if device is not None:
        A = A.to(device)
    return A


# =============== 区域路由注意力核心函数 ===============

def regional_routing_attention_torch(
    query: Tensor, key: Tensor, value: Tensor, scale: float,
    region_graph: LongTensor, region_size: Tuple[int, int],
    kv_region_size: Optional[Tuple[int, int]] = None,
    auto_pad: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    功能：基于区域图的路由注意力 + 二次稀疏选择
    输入:
        query, key, value: Q/K/V 特征
        region_graph: 区域图（top-k 的索引）
        region_size: 区域大小
    输出:
        output: 注意力加权结果 (B, C, H, W)
        attn: token-to-token 注意力得分
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # ---- 步骤1：必要时补零填充 ----
    q_pad_b = q_pad_r = kv_pad_b = kv_pad_r = 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if q_pad_b > 0 or q_pad_r > 0:
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if kv_pad_b > 0 or kv_pad_r > 0:
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))

    # ---- 步骤2：图像转序列 ----
    query, q_region_h, q_region_w = _grid2seq(query, region_size, nhead)
    key, _, _ = _grid2seq(key, kv_region_size, nhead)
    value, _, _ = _grid2seq(value, kv_region_size, nhead)

    # ---- 步骤3：根据区域图收集 K/V ----
    bs, nhead, kv_nregion, kv_reg_sz, head_dim = key.size()
    graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).expand(-1, -1, -1, -1, kv_reg_sz, head_dim)

    key_g = torch.gather(
        key.view(bs, nhead, 1, kv_nregion, kv_reg_sz, head_dim).expand(-1, -1, q_nregion, -1, -1, -1),
        dim=3, index=graph
    )
    value_g = torch.gather(
        value.view(bs, nhead, 1, kv_nregion, kv_reg_sz, head_dim).expand(-1, -1, q_nregion, -1, -1, -1),
        dim=3, index=graph
    )

    # ---- 步骤4：token-to-token 注意力 ----
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)

    # ---- 步骤5：像素级稀疏选择（再选1/8） ----
    keep = max(1, (topk * kv_reg_sz) // 8)
    score, index = attn.topk(keep, dim=-1)

    v_g_un = value_g.flatten(-3, -2).unsqueeze(-3).expand(-1, -1, -1, attn.size(3), -1, -1)
    idx = index.unsqueeze(-1).expand(-1, -1, -1, -1, -1, head_dim)
    v_g_select = torch.gather(v_g_un, dim=4, index=idx)

    a_g = torch.softmax(score.unsqueeze(-2), dim=-1)
    output = (a_g @ v_g_select).squeeze(-2)

    # ---- 步骤6：恢复成图像形式 ----
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # 去掉 padding
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


# =============== STAR-DSSA 模块 ===============

class SEGate(nn.Module):
    """SE 通道门控，用于给局部卷积增强加权"""
    def __init__(self, c, r=4):
        super().__init__()
        hidden = max(8, c // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, c, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)


class STAR_DSSA(nn.Module):
    """
    STAR-DSSA 模块
    (Saliency & Topology Aware Routed Dual-Selective Self-Attention)
    - 显著性引导：区域 Q/K 由显著性权重加权池化得到
    - 拓扑感知：在区域图中加入邻域先验，提升空间一致性
    - 双稀疏选择：区域级 + 像素级两次筛选，降低复杂度
    - 门控局部增强：5x5 深度卷积 + SE 门控，补充细节
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4,
                 lce_kernel=5, topo_radius=1, topo_weight_init=0.5, auto_pad=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (dim ** -0.5)

        # QKV 投影
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # 显著性分支，用于区域加权池化
        self.saliency = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

        # 局部上下文增强 + 通道门控
        self.lce = nn.Conv2d(dim, dim, kernel_size=lce_kernel, padding=lce_kernel // 2, groups=dim) if lce_kernel > 0 else None
        self.lce_gate = SEGate(dim) if lce_kernel > 0 else None

        # 路由参数
        self.topk = topk
        self.n_win = n_win
        self.auto_pad = auto_pad

        # 可学习的拓扑混合权重
        self.topo_logit = nn.Parameter(torch.tensor(float(topo_weight_init)).log() - torch.tensor(1. - float(topo_weight_init)).log())
        self.topo_radius = topo_radius

    @staticmethod
    def _region_weighted_pool(x: Tensor, weight_map: Tensor, region_size: Tuple[int, int], detach=True):
        """
        区域加权池化：利用显著性权重计算区域级特征
        """
        if detach:
            x = x.detach()
            weight_map = weight_map.detach()

        wm = F.softplus(weight_map) + 1e-6
        rh, rw = region_size
        num = F.avg_pool2d(x * wm, kernel_size=(rh, rw), ceil_mode=True, count_include_pad=False) * (rh * rw)
        den = F.avg_pool2d(wm, kernel_size=(rh, rw), ceil_mode=True, count_include_pad=False) * (rh * rw)
        pooled = num / (den + 1e-6)
        B, C, Rh, Rw = pooled.shape
        return pooled.permute(0, 2, 3, 1).reshape(B, Rh * Rw, C)

    def forward(self, x: Tensor, ret_attn_mask: bool = False):
        B, C, H, W = x.shape
        rh = H // self.n_win
        rw = W // self.n_win
        region_size = (max(1, rh), max(1, rw))

        # ---- 步骤1：QKV 分解 ----
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # ---- 步骤2：显著性加权区域池化 ----
        sal = self.saliency(x)
        q_r = self._region_weighted_pool(q, sal, region_size)
        k_r = self._region_weighted_pool(k, sal, region_size)

        # ---- 步骤3：区域相似度矩阵 ----
        a_r = q_r @ k_r.transpose(1, 2)

        # ---- 步骤4：加入拓扑偏置 ----
        Rh = max(1, H // region_size[0])
        Rw = max(1, W // region_size[1])
        A = _build_adjacency_bias(Rh, Rw, radius=self.topo_radius, device=x.device)
        topo_w = torch.sigmoid(self.topo_logit)
        a_r = (1.0 - topo_w) * a_r + topo_w * (a_r + A)

        # ---- 步骤5：top-k 区域路由 ----
        R = a_r.size(1)
        topk = min(self.topk, R)
        _, idx_r = torch.topk(a_r, k=topk, dim=-1)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # ---- 步骤6：区域路由注意力 ----
        out, attn_pix = regional_routing_attention_torch(
            query=q, key=k, value=v, scale=self.scale,
            region_graph=idx_r, region_size=region_size,
            kv_region_size=region_size, auto_pad=self.auto_pad
        )

        # ---- 步骤7：局部卷积增强 ----
        if self.lce is not None:
            lce_feat = self.lce(v)
            lce_feat = self.lce_gate(lce_feat)
            out = out + lce_feat

        out = self.proj(out)

        if ret_attn_mask:
            return out, attn_pix
        return out


# =============== 测试代码 ===============
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 1, 32, 128, 128
    nhead = 8
    n_win = 16
    topk = 6

    x = torch.randn(B, C, H, W)
    star_dssa = STAR_DSSA(dim=C, num_heads=nhead, n_win=n_win, topk=topk)

    y = star_dssa(x)
    print(star_dssa)
    print("输入形状:", x.shape)
    print("\n微信公众号:CV缝合救星\n")
    print("输出形状:", y.shape)
    print("✅ STAR-DSSA 前向传播成功")
