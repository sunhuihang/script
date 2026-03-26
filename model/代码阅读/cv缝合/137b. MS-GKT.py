import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================
# 简化版 KANLinear 占位符（可替换为你自己的实现）
# =========================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        # 这里简单用 Linear 代替，保证代码可跑
        # 如果你有真正的 KAN 实现，可直接替换为 from kan import KANLinear
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# =========================
# 工具函数：截断正态初始化
# =========================
def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor

# =========================
# 组内通道 KAN 映射模块（对应论文 GKT 中的组内 Φ(g)）
# 输入/输出：x ∈ [B, N, C]
# 按通道维度分组，每组单独走一个 KANLinear
# =========================
class GroupKANChannelBlock(nn.Module):
    def __init__(self, in_channels, group=16, grid_size=5, spline_order=3):
        super(GroupKANChannelBlock, self).__init__()
        assert in_channels % group == 0, "in_channels 必须能被 group 整除"

        self.in_channels = in_channels
        self.group = group
        self.ch_per_group = in_channels // group

        # 每个通道组对应一个 KANLinear，用于建模组内通道间的非线性关系
        self.group_kan = nn.ModuleList([
            KANLinear(self.ch_per_group, self.ch_per_group,
                      grid_size=grid_size, spline_order=spline_order)
            for _ in range(group)
        ])

    def forward(self, x):
        """
        x: [B, N, C]
        return: [B, N, C]
        """
        B, N, C = x.shape
        group_outputs = []

        # 按通道分组进行组内 KAN 变换
        for g in range(self.group):
            ch_start = g * self.ch_per_group
            ch_end = (g + 1) * self.ch_per_group

            # 取出当前组的特征 [B, N, ch_per_group]
            x_g = x[:, :, ch_start:ch_end]
            # 展平成 [B * N, ch_per_group] 以匹配 KANLinear 的输入
            x_g = x_g.reshape(B * N, self.ch_per_group)

            # 组内 KAN 映射
            kan_out = self.group_kan[g](x_g)        # [B * N, ch_per_group]
            kan_out = kan_out.view(B, N, self.ch_per_group)

            group_outputs.append(kan_out)

        # 拼回完整通道维度 [B, N, C]
        x_out = torch.cat(group_outputs, dim=2)
        return x_out

# =========================
# PW + DW 卷积：负责跨组与空间交互
# 输入/输出：x ∈ [B, N, C]
# 内部短暂还原为 [B, C, H, W] 进行卷积，再展平回 token 形式
# =========================
class PWDWConv(nn.Module):
    def __init__(self, dim=768, expansion=1):
        super(PWDWConv, self).__init__()
        hidden_dim = dim * expansion

        # 1x1 点卷积：负责跨通道（跨组）混合
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 深度可分卷积：负责空间局部交互与建模
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                                bias=True, groups=hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        """
        x: [B, N, C]，其中 N = H * W
        return: [B, N, C]
        """
        B, N, C = x.shape
        assert N == H * W, "N 必须等于 H * W 才能正确 reshape 为特征图"

        # token 形式 [B, N, C] -> 特征图形式 [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)

        # 点卷积 + BN + ReLU
        x = self.pwconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 深度卷积 + BN + ReLU
        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 特征图形式还原回 token 形式 [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        return x

# =========================
# 纯 GKT 核心模块（无 GKA）
# - 输入：x [B, N, C]，外加 H, W（N = H * W）
# - 输出：same shape [B, N, C]
# - 结构：多层 (GroupKANChannelBlock + PWDWConv) 残差堆叠
# =========================
class GroupKANTransform(nn.Module):
    def __init__(self,
                 dim,
                 num_gkt_layers=3,
                 group_kan_num=16,
                 drop=0.0):
        super().__init__()

        assert dim % group_kan_num == 0, "dim 必须能被 group_kan_num 整除"

        self.dim = dim
        self.group_kan_num = group_kan_num
        self.drop = nn.Dropout(drop)

        # 多层 GKT：每层包含组内通道 KAN + PW/DW Conv
        self.gkt_fc = nn.ModuleList([
            GroupKANChannelBlock(dim, group=self.group_kan_num)
            for _ in range(num_gkt_layers)
        ])
        self.gkt_conv = nn.ModuleList([
            PWDWConv(dim) for _ in range(num_gkt_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        # 模块参数初始化：Linear / Conv / LayerNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        x: [B, N, C]，N = H * W
        return: [B, N, C]
        """
        for fc, conv in zip(self.gkt_fc, self.gkt_conv):
            residual = x                      # 残差连接
            x = fc(x)                         # 组内 KAN 通道变换
            x = conv(x, H, W)                 # 空间 + 跨组卷积交互
            x = residual + self.drop(x)       # 残差叠加 + dropout
        return x

# ============================================================
# ⭐ 魔改创新模块：MS-GKT
# Multi-Scale Grouped Kolmogorov Transform Block
# 多尺度分组 Kolmogorov 变换块
#
# 设计思路：
#   1）保持原始 GKT 路径（GroupKANTransform），负责“功能性通道非线性 + 空间建模”
#   2）新增本地卷积分支：在原分辨率上用轻量卷积捕获局部纹理与边缘细节
#   3）引入通道注意力门控：根据全局语义自适应调节两条分支的贡献
#   4）整体残差形式输出：输出 = 输入 + gate_kan * F_kan + gate_loc * F_local
#   5）输入/输出尺寸不变：[B, C, H, W] -> [B, C, H, W]
# ============================================================
class MultiScaleGroupKANTransformCV(nn.Module):
    """
    MS-GKT：Multi-Scale Grouped Kolmogorov Transform Block
    多尺度分组 KAN 变换块

    适用场景：
        - 可作为 U-Net / FPN / SegFormer / ConvNeXt / Mamba 等骨干中的
          “功能性替换块”或“增强块”，直接插在 BCHW 流程中使用。
    """
    def __init__(self,
                 dim,
                 num_gkt_layers=3,
                 group_kan_num=16,
                 drop=0.0,
                 local_expansion=1):
        super().__init__()
        self.dim = dim

        # ① 原始 GKT 路径（token 形式的 GroupKANTransform）
        self.kan_core = GroupKANTransform(
            dim=dim,
            num_gkt_layers=num_gkt_layers,
            group_kan_num=group_kan_num,
            drop=drop
        )

        # ② 本地卷积分支（完全在 BCHW 空间域操作）
        #    使用深度可分卷积捕获局部结构 + 1x1 卷积做通道重映射
        hidden_dim = dim * local_expansion
        self.local_conv = nn.Sequential(
            # 深度卷积：只在空间上卷积，每个通道独立
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            # 1x1 卷积：混合通道信息
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 映射回原始通道数
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        # ③ 通道注意力门控：自适应融合 GKT 路径与本地卷积分支
        #    输入：全局池化后的通道描述向量 [B, C]
        #    输出：两个通道门控向量 [B, 2*C]，分别对应 GKT 分支和本地分支
        reduction = max(1, dim // 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, dim * 2)  # 2*C，前 C 维给 GKT 分支，后 C 维给 local 分支
        )

        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # 初始化所有子模块参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.dim, "输入通道数必须与 dim 一致"

        # ---------- 分支一：GKT（KAN + 卷积）路径 ----------
        # BCHW -> BNC（token 形式）
        x_tok = x.flatten(2).transpose(1, 2)        # [B, N, C], N=H*W
        kan_tok = self.kan_core(x_tok, H, W)        # [B, N, C]
        # BNC -> BCHW
        kan_feat = kan_tok.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        # ---------- 分支二：本地卷积分支 ----------
        local_feat = self.local_conv(x)             # [B, C, H, W]

        # ---------- 通道注意力门控 ----------
        # 使用输入 x 做全局平均池化，得到图像级通道描述
        # pool: [B, C, 1, 1] -> [B, C]
        pool = F.adaptive_avg_pool2d(x, output_size=1).view(B, C)

        # MLP 生成两个分支的通道权重 [B, 2*C]
        gates = self.channel_mlp(pool)              # [B, 2*C]
        gates = self.sigmoid(gates)                 # 归一化到 [0,1]

        # 拆分为 GKT 分支门控和 local 分支门控：[B, C] + [B, C]
        gate_kan, gate_local = gates.chunk(2, dim=1)
        gate_kan = gate_kan.view(B, C, 1, 1)
        gate_local = gate_local.view(B, C, 1, 1)

        # ---------- 分支融合 + 残差输出 ----------
        # 输出 = 原始输入 + gated_GKT + gated_local
        out = x + gate_kan * kan_feat + gate_local * local_feat

        return out

# =========================
# 主函数：测试 MS-GKT 模块
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 1
    C = 128
    H = 32
    W = 32

    # 构造随机输入
    x = torch.randn(B, C, H, W)

    # 实例化 MS-GKT 模块
    msgkt = MultiScaleGroupKANTransformCV(
        dim=C,
        num_gkt_layers=3,
        group_kan_num=16,
        drop=0.0,
        local_expansion=1
    )

    print("===== MS-GKT MultiScaleGroupKANTransformCV 模块结构 =====")
    print(msgkt)

    # 前向传播
    y = msgkt(x)

    print("\n===== 输入 / 输出尺寸检查 =====")
    print("输入 x 形状:", x.shape)
    print("输出 y 形状:", y.shape)

    print("\n✨ MS-GKT 多尺度分组 Kolmogorov 变换块 · CV 缝合救星 · 独家魔改复现 Done!")
