import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.nn.init import trunc_normal_

# 工具函数：保证输入是二元组
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

# ---------------- LayerNorm2d ----------------
class LayerNorm2d(nn.Module):
    """
    二维归一化层（自定义 LayerNorm）
    - 对输入的 [B, C, H, W] 按通道维度做归一化
    - 保持空间信息不变
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))  # 可学习缩放参数
        self.bias = nn.Parameter(torch.zeros(channels))   # 可学习偏移参数
        self.eps = eps                                    # 防止除零的小常数

    def forward(self, x):
        # 计算均值和方差 (按通道求均值)
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # 归一化
        y = (x - mu) / (var + self.eps).sqrt()
        # 应用缩放和平移
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


# ---------------- HDNA 模块 ----------------
class HDNA(nn.Module):
    """
    HDNA 模块 (Hybrid Deformable Neighborhood Attention)
    核心创新点：
    1. 局部邻域注意力 + 全局通道注意力 (混合建模)
    2. 层次可变形偏移：小卷积预测精细偏移，大卷积预测粗粒度偏移，融合得到最终 offset
    3. 保留相对位置偏置和深度卷积位置编码
    """
    def __init__(
        self,
        dim: int,           # 输入通道数
        num_heads: int,     # 注意力头数
        kernel_size: int,   # 邻域大小 (窗口大小)
        dilation: int = 1,  # 空洞卷积扩张率
        offset_range_factor=1.0,  # 偏移范围控制系数
        stride=1,           # 步幅（默认为1）
        use_pe=True,        # 是否使用位置编码
        rel_pos_bias=True,  # 是否使用相对位置偏置
        attn_drop=0.0,      # 注意力 dropout
        proj_drop=0.0,      # 输出投影 dropout
    ):
        super().__init__()
        assert dim % num_heads == 0, "通道数必须能被注意力头数整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads          # 每个 head 的维度
        self.scale = self.head_dim ** -0.5        # 缩放因子 (防止 softmax 前数值过大)
        self.dim = dim
        self.ksize = kernel_size
        self.dilation = dilation
        self.offset_range_factor = offset_range_factor

        # ---------------- 层次可变形偏移预测器 ----------------
        # 小卷积 (3x3) -> 学习精细的局部偏移
        self.conv_offset_small = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1, 1, 0, bias=False),
        )
        # 大卷积 (7x7) -> 学习粗粒度的全局偏移
        self.conv_offset_large = nn.Sequential(
            nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1, 1, 0, bias=False),
        )

        # ---------------- QKV 投影 ----------------
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_k = nn.Conv2d(dim, dim, 1)
        self.proj_v = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)

        # ---------------- 相对位置偏置 ----------------
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, 2*kernel_size-1, 2*kernel_size-1)
            )
            trunc_normal_(self.rpb, std=0.02)  # 初始化为小数值
        else:
            self.register_parameter("rpb", None)

        # ---------------- 深度卷积位置编码 ----------------
        self.rpe_table = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # ---------------- 全局通道注意力 (SE-like) ----------------
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # 全局池化
            nn.Conv2d(dim, dim//4, 1),          # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//4, dim, 1),          # 升维
            nn.Sigmoid()                        # 输出通道权重
        )

        # 预计算邻域内索引，用于相对位置偏置
        self.register_buffer("rpb_index", self._build_rpb_index(), persistent=False)

    def _build_rpb_index(self):
        """
        构建一个邻域索引表，用于从 rpb 参数中取出对应的偏置
        """
        k = self.ksize
        cen = k // 2
        idx = []
        for dy in range(-cen, cen+1):
            for dx in range(-cen, cen+1):
                idx.append((dy, dx))
        idx_map = torch.tensor(idx, dtype=torch.long)
        idx_map[:,0] += (k-1)
        idx_map[:,1] += (k-1)
        return idx_map

    def forward(self, x):
        """
        前向传播
        输入: x [B, C, H, W]
        输出: y [B, C, H, W]
        """
        B, C, H, W = x.shape
        k, P = self.ksize, self.ksize*self.ksize
        pad = (k//2) * self.dilation

        # ---------------- 计算层次偏移 ----------------
        off_s = self.conv_offset_small(x)   # 精细偏移
        off_l = self.conv_offset_large(x)   # 粗偏移
        offset = torch.tanh(off_s + off_l) * self.offset_range_factor  # 融合后的偏移

        # ---------------- 可变形采样 ----------------
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1,1,H,device=x.device),
            torch.linspace(-1,1,W,device=x.device),
            indexing="ij"
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [H,W,2]
        grid = base_grid.unsqueeze(0).expand(B,-1,-1,-1) + offset.permute(0,2,3,1)
        x_sampled = F.grid_sample(x, grid, mode="bilinear", align_corners=True)

        # ---------------- QKV 投影 ----------------
        q = self.proj_q(x)
        k_map = self.proj_k(x_sampled)
        v_map = self.proj_v(x_sampled)

        # 将 Q reshape 成 [B, num_heads, N, head_dim]
        q_heads = q.view(B, self.num_heads, self.head_dim, H, W)
        q_flat = q_heads.permute(0,1,3,4,2).reshape(B,self.num_heads,H*W,self.head_dim)
        q_flat = q_flat * self.scale

        # unfold 提取邻域 patch，用作 K/V
        k_unf = F.unfold(k_map, k, dilation=self.dilation, padding=pad)  # [B,C*P,N]
        v_unf = F.unfold(v_map, k, dilation=self.dilation, padding=pad)
        k_unf = k_unf.view(B,self.num_heads,self.head_dim,P,H*W).permute(0,1,4,3,2)
        v_unf = v_unf.view(B,self.num_heads,self.head_dim,P,H*W).permute(0,1,4,3,2)

        # ---------------- 局部注意力 ----------------
        attn = (q_flat.unsqueeze(3) * k_unf).sum(-1)  # [B,h,N,P]
        if self.rpb is not None:  # 加入相对位置偏置
            idx = self.rpb_index
            rpb_flat = self.rpb[:, idx[:,0], idx[:,1]]  # [h,P]
            attn = attn + rpb_flat.unsqueeze(0).unsqueeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 聚合 V
        out_flat = (attn.unsqueeze(-1) * v_unf).sum(3)  # [B,h,N,dh]

        # reshape 回原图 [B,C,H,W]
        out = out_flat.view(B,self.num_heads,H,W,self.head_dim).permute(0,1,4,2,3).reshape(B,C,H,W)

        # ---------------- 位置编码 ----------------
        out = out + self.rpe_table(q)

        # ---------------- 全局通道注意力 ----------------
        g = self.global_attn(out)
        out = out * g

        # 输出投影
        y = self.proj_drop(self.proj_out(out))
        return y


# ---------------- Quick test ----------------
if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128)
    attn = HDNA(
        dim=64, num_heads=8, kernel_size=7,
        rel_pos_bias=True
    )
    with torch.no_grad():
        y = attn(x)
    print(attn)
    print("\n微信公众号:CV缝合救星\n")
    print("输入:", x.shape, "输出:", y.shape)
