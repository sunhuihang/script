import torch
import torch.nn as nn
from torch import Tensor

# -----------------------------
# 基础半卷积：对一半通道做3x3卷积，另一半通道原样保留
# -----------------------------
class HalfConv(nn.Module):
    def __init__(self, dim: int, n_div: int = 2, kernel_size: int = 3):
        """
        dim: 输入通道数
        n_div: 将通道切为 n_div 份，这里用 2 表示一半做卷积、一半不动
        kernel_size: 半卷积使用的卷积核尺寸（默认3）
        """
        super().__init__()
        assert n_div >= 2, "n_div 至少为 2 才有“半卷积”之意"
        self.dim_conv = dim // n_div               # 需要卷积的通道数
        self.dim_skip = dim - self.dim_conv        # 保留不变的通道数
        padding = kernel_size // 2                 # 保持特征图尺寸不变
        # 仅对前半部分做标准卷积（这里不使用BN，保持轻量与通用）
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv,
                                      kernel_size=kernel_size, stride=1,
                                      padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # 按通道拆分为 [卷积部分, 跳过部分]
        x_conv, x_skip = torch.split(x, [self.dim_conv, self.dim_skip], dim=1)
        # 只对“卷积部分”做3x3卷积
        x_conv = self.partial_conv(x_conv)
        # 拼接回去，形成“半卷积”的输出
        out = torch.cat((x_conv, x_skip), dim=1)
        return out


# -----------------------------
# 工具函数：Channel Shuffle
# 作用：在组间打散通道，促进跨组的信息交互
# 若通道数不能被 groups 整除，则退化为不shuffle（保证鲁棒性）
# -----------------------------
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    channels_per_group = c // groups
    # 形状变换：B, (g * cpg), H, W -> B, g, cpg, H, W
    x = x.view(b, groups, channels_per_group, h, w)
    # 交换组维与通道内维：B, cpg, g, H, W
    x = x.transpose(1, 2).contiguous()
    # 展平回原通道维度：B, C, H, W
    x = x.view(b, c, h, w)
    return x


# -----------------------------
# 轻量级 SE 门控（Squeeze-Excitation）
# 作用：利用全局上下文自适应地重标定各通道重要性
# -----------------------------
class SEGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        """
        dim: 通道数
        reduction: 降维比（越大代表越轻量）
        """
        super().__init__()
        hidden = max(dim // reduction, 4)  # 保底不小于4，避免极小通道数时退化
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，得到通道级描述
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)     # 较平滑的激活，训练更稳定
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w                          # 通道重标定


# -----------------------------
# 深度可分离卷积（Depthwise Separable）
# 作用：在保持计算量极低的前提下做空间/通道混合
# -----------------------------
class DWConvPW(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        """
        dim: 通道数（深度卷积和逐点卷积的输入/输出通道）
        kernel_size: 深度卷积核大小
        """
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                            stride=1, padding=padding, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()  # 轻量非线性提升表达

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        return x


# -----------------------------
# GS-HalfConv（Gated Shuffle Half-Convolution）主模块
# 设计要点：
# 1) 通道三分组：每组做 HalfConv（仅一半通道卷积）
# 2) Channel Shuffle：跨组打散通道，弥补“分组独立”的信息孤岛
# 3) 深度可分离卷积：轻量空间混合
# 4) SE 门控：全局自适应通道重标定
# 5) 残差 + 可学习缩放：稳定训练，便于堆叠
# -----------------------------
class GSHalfConv(nn.Module):
    def __init__(self, dim: int, groups: int = 3,
                 n_div: int = 2, se_reduction: int = 4,
                 shuffle_groups: int = 3, dw_kernel: int = 3):
        """
        dim: 输入/输出通道数
        groups: 通道分组数量（论文默认3组）
        n_div: HalfConv中“卷积份数”划分（默认2 -> 一半卷积）
        se_reduction: SE门控的降维比
        shuffle_groups: channel shuffle 的分组数（通常与 groups 相同）
        dw_kernel: 深度卷积核大小（3/5均可）
        """
        super().__init__()
        assert groups >= 1, "groups 至少为 1"

        # 计算每组通道数，余数并入最后一组，保证总通道数不变
        base = dim // groups
        remainder = dim - base * (groups - 1)
        group_dims = [base] * (groups - 1) + [remainder]

        # 为每一组构建 HalfConv
        self.group_blocks = nn.ModuleList([
            HalfConv(d, n_div=n_div, kernel_size=3) for d in group_dims
        ])

        # 记录一些元信息
        self.groups = groups
        self.shuffle_groups = shuffle_groups if (dim % shuffle_groups == 0) else 1

        # 轻量空间/通道混合：深度可分离卷积
        self.mix = DWConvPW(dim, kernel_size=dw_kernel)

        # 全局门控：SE
        self.se = SEGate(dim, reduction=se_reduction)

        # 残差缩放参数：初始化为一个较小值，有助于稳定训练
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # 简单的归一化层（可选）：在轻量模型中常用 LayerNorm2d/BN
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 1) 通道三分组：按计算好的每组通道数拆分
        #   注意：最后一组带上余数，兼容任意 dim
        splits = []
        start = 0
        for blk in self.group_blocks:
            d = blk.partial_conv.in_channels if isinstance(blk, HalfConv) else None  # 占位，无实际用
            # 由于上面的占位不可用，这里用更稳妥的做法：从模块参数反推通道数
            # HalfConv 无法直接拿到“全组通道数”，改为记录 split 大小：
            # 解决方案：重写 group_dims 存在模块中，或者在 ModuleList 外部保存
            # 为简洁，这里直接重新计算 group_dims
            pass

        # 为了简洁与鲁棒，这里重新计算 group_dims（与 __init__ 同逻辑）
        base = x.shape[1] // self.groups
        remainder = x.shape[1] - base * (self.groups - 1)
        group_dims = [base] * (self.groups - 1) + [remainder]

        xs = torch.split(x, group_dims, dim=1)

        # 2) 每组做 HalfConv（仅一半通道卷积）
        outs = []
        for t, blk in zip(xs, self.group_blocks):
            outs.append(blk(t))

        # 3) 组级拼接
        out = torch.cat(outs, dim=1)

        # 4) Channel Shuffle：跨组打散通道（若通道数不可整除则自动跳过）
        out = channel_shuffle(out, self.shuffle_groups)

        # 5) 轻量空间/通道混合（深度可分离卷积）
        out = self.mix(out)

        # 6) 全局通道门控（SE）
        out = self.se(out)

        # 7) 归一化 + 残差（带可学习缩放）
        out = self.norm(out)
        out = identity + self.res_scale * out
        return out


# -----------------------------
# 兼容你原始 CGHalfConv 的外层包装（可选）
# 若你希望直接替换 CGHalfConv，用下方类名即可
# -----------------------------
class CGHalfConv_GS(nn.Module):
    """
    用 GS-HalfConv 替换原 CGHalfConv 内部的三组 HalfConv，并额外引入
    shuffle + 深度可分离卷积 + SE 门控 + 残差缩放。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.block = GSHalfConv(dim=dim, groups=3, n_div=2,
                                se_reduction=4, shuffle_groups=3, dw_kernel=3)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# -----------------------------
# 简单的单元测试：可直接运行
# -----------------------------
if __name__ == "__main__":
    # 输入张量 [B, C, H, W]
    x = torch.randn(1, 32, 256, 256)

    # 初始化 GS-HalfConv 模块
    gshc = GSHalfConv(dim=32)

    # 前向传播
    y = gshc(x)

    print("\n微信公众号:CV缝合救星\n")
    print(gshc)
    print("\n输入形状:", x.shape)
    print("输出形状:", y.shape)

    # 统计参数量（单位：百万）
    total_params = sum(p.numel() for p in gshc.parameters()) / 1e6
    print(f"参数量: {total_params:.3f} M")
