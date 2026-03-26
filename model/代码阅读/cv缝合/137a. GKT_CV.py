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
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# =========================
# 工具函数
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
# 组内通道 KAN 映射（论文 GKT 中的组内 Φ(g)）
# 输入/输出：[B, N, C]
# =========================
class GroupKANChannelBlock(nn.Module):
    def __init__(self, in_channels, group=16, grid_size=5, spline_order=3):
        super(GroupKANChannelBlock, self).__init__()
        assert in_channels % group == 0, "in_channels must be divisible by group"

        self.in_channels = in_channels
        self.group = group
        self.ch_per_group = in_channels // group

        self.group_kan = nn.ModuleList([
            KANLinear(self.ch_per_group, self.ch_per_group,
                      grid_size=grid_size, spline_order=spline_order)
            for _ in range(group)
        ])

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        group_outputs = []
        for g in range(self.group):
            ch_start = g * self.ch_per_group
            ch_end = (g + 1) * self.ch_per_group
            x_g = x[:, :, ch_start:ch_end]          # [B, N, ch_per_group]
            x_g = x_g.reshape(B * N, self.ch_per_group)
            kan_out = self.group_kan[g](x_g)        # [B*N, ch_per_group]
            kan_out = kan_out.view(B, N, self.ch_per_group)
            group_outputs.append(kan_out)
        x_out = torch.cat(group_outputs, dim=2)      # [B, N, C]
        return x_out

# =========================
# PW+DW 卷积：负责跨组和空间交互
# 输入/输出：[B, N, C]
# =========================
class PWDWConv(nn.Module):
    def __init__(self, dim=768, expansion=1):
        super(PWDWConv, self).__init__()
        hidden_dim = dim * expansion

        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                                bias=True, groups=hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        # x: [B, N, C] 其中 N = H * W
        B, N, C = x.shape
        assert N == H * W, "N must be equal to H*W for reshape"

        x = x.transpose(1, 2).view(B, C, H, W)   # [B, C, H, W]

        x = self.pwconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.flatten(2).transpose(1, 2)         # [B, N, C]
        return x

# =========================
# 纯 GKT 模块（无 GKA）
# - 输入：x [B, N, C]，外加 H, W（N = H*W）
# - 输出：same shape [B, N, C]
# - 结构：多层（GroupKANChannelBlock + PWDWConv）残差堆叠
# =========================
class GroupKANTransform(nn.Module):
    def __init__(self,
                 dim,
                 num_gkt_layers=3,
                 group_kan_num=16,
                 drop=0.0):
        super().__init__()

        assert dim % group_kan_num == 0, "dim must be divisible by group_kan_num"

        self.dim = dim
        self.group_kan_num = group_kan_num
        self.drop = nn.Dropout(drop)

        # 多层 GKT：组内通道 KAN + PWDWConv
        self.gkt_fc = nn.ModuleList([
            GroupKANChannelBlock(dim, group=self.group_kan_num)
            for _ in range(num_gkt_layers)
        ])
        self.gkt_conv = nn.ModuleList([
            PWDWConv(dim) for _ in range(num_gkt_layers)
        ])

        self._init_weights()

    def _init_weights(self):
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
        x: [B, N, C],  N = H * W
        return: [B, N, C]
        """
        # 多层 GKT：组内 KAN + PW/DW conv（带残差）
        for fc, conv in zip(self.gkt_fc, self.gkt_conv):
            residual = x
            x = fc(x)                # GroupKANChannelBlock, [B, N, C]
            x = conv(x, H, W)        # PWDWConv, [B, N, C]
            x = residual + self.drop(x)

        return x

# =========================
# CV 版纯 GKT 模块
# - 输入：x [B, C, H, W]
# - 输出：same shape [B, C, H, W]
# - 内部：BCHW -> BNC -> GroupKANTransform -> BCHW
# =========================
class GroupKANTransformCV(nn.Module):
    def __init__(self,
                 dim,
                 num_gkt_layers=3,
                 group_kan_num=16,
                 drop=0.0):
        super().__init__()
        self.dim = dim
        self.core = GroupKANTransform(
            dim=dim,
            num_gkt_layers=num_gkt_layers,
            group_kan_num=group_kan_num,
            drop=drop
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.dim, "Channel dim must match `dim` in GroupKANTransformCV"

        N = H * W
        # BCHW -> BNC
        x_tok = x.flatten(2).transpose(1, 2)   # [B, N, C]

        # 走纯 GKT
        y_tok = self.core(x_tok, H, W)        # [B, N, C]

        # BNC -> BCHW
        y = y_tok.transpose(1, 2).view(B, C, H, W)
        return y

# =========================
# 主函数：测试 CV 版纯 GKT 模块
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 1
    C = 128
    H = 16
    W = 16

    x = torch.randn(B, C, H, W)

    gkt_cv = GroupKANTransformCV(
        dim=C,
        num_gkt_layers=3,
        group_kan_num=16,
        drop=0.0
    )

    print("===== CV 版 纯 GKT GroupKANTransformCV 模型结构 =====")
    print(gkt_cv)

    y = gkt_cv(x)

    print("\n===== 输入 / 输出尺寸检查 =====")
    print("输入 x 形状:", x.shape)
    print("输出 y 形状:", y.shape)

    print("\n✨ CV 缝合救星 · 纯 GKT GroupKAN BCHW 即插即用版 · 独家复现 Done!")
