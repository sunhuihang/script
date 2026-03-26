import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


# 点注意力模块（Point Attention）
class PointAttention(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            norm_layer(dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


# 局部注意力模块（Local Attention）
class LocalAttention(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        return self.conv(x)


# 中程感知模块（Medium-Range Attention）
class MediumRangeAttention(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            norm_layer(dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# 拓扑感知全局注意力模块（Topology-Aware Attention）
class TopologyAwareAttention(nn.Module):
    def __init__(self, dim, head_dim=4):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.randn(1, self.num_heads, 1, 1))  # 可学习的拓扑偏置项

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # 展平空间维度
        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.rel_bias  # 加拓扑偏置
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


# 边缘引导模块（Edge Guidance）
class EdgeGuidance(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        edge_map = self.edge_conv(x)
        return torch.sigmoid(edge_map)  # 输出边缘图


# 动态加权融合模块（根据血管粗细进行融合）
class DynamicWeightFusion(nn.Module):
    def __init__(self, num_inputs, in_channels):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_inputs * in_channels, num_inputs, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.num_inputs = num_inputs

    def forward(self, inputs):
        B, C, H, W = inputs[0].shape
        concat = torch.cat(inputs, dim=1)  # 拼接所有输入
        weights = self.weight_gen(concat).view(B, self.num_inputs, 1, 1, 1)  # 得到权重
        stacked = torch.stack(inputs, dim=1)  # 堆叠成 [B, N, C, H, W]
        fused = (weights * stacked).sum(dim=1)  # 权重融合
        return fused


# 主模块：ETFAttn（结构感知 + 拓扑建模 + 动态融合）
class ETFAttn(nn.Module):
    def __init__(self, dim=32, mlp_ratio=4.0, drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.dim_split = dim // 4  # 每个注意力分支通道数为 dim/4

        # 各分支注意力机制
        self.pa = PointAttention(self.dim_split, norm_layer, act_layer)      # 点注意力
        self.la = LocalAttention(self.dim_split, norm_layer, act_layer)      # 局部注意力
        self.mra = MediumRangeAttention(self.dim_split, norm_layer)          # 中程注意力
        self.gatt = TopologyAwareAttention(self.dim_split)                   # 全局拓扑注意力

        # 辅助模块
        self.edge_guidance = EdgeGuidance(self.dim_split)                    # 边缘引导
        self.dynamic_fusion = DynamicWeightFusion(4, self.dim_split)         # 动态融合
        self.reproject = nn.Conv2d(self.dim_split, dim, kernel_size=1)       # 通道升维

        self.norm = norm_layer(dim)                                          # 总体归一化
        self.mlp = nn.Sequential(                                            # FFN前馈网络
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            act_layer(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x  # 残差连接
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)  # 通道切分

        # x1：边缘引导的点注意力
        edge_map = self.edge_guidance(x1)
        x1 = self.pa(x1) * edge_map

        # x2：局部注意力
        x2 = self.la(x2)

        # x3：中程注意力
        x3 = self.mra(x3)

        # x4：拓扑感知注意力
        x4 = self.gatt(x4)

        # 融合 + 升维 + FFN
        fused = self.dynamic_fusion([x1, x2, x3, x4])
        fused = self.reproject(fused)
        out = shortcut + self.drop_path(self.mlp(self.norm(fused)))
        return out


# 测试代码
if __name__ == "__main__":
    model = ETFAttn(dim=32).cuda()
    dummy = torch.randn(2, 32, 64, 64).cuda()
    out = model(dummy)
    print("For Perry!")
    print("输入形状:", dummy.shape)
    print("输出形状:", out.shape)
