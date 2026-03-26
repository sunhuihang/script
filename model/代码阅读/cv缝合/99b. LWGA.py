import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


# ==========================================
# PA模块：Point Attention 点注意力
# ==========================================
class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            norm_layer(dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)
        return x


# ==========================================
# LA模块：Local Attention 局部注意力
# ==========================================
class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# ==========================================
# MRA模块：Medium-Range Attention 中距离注意力
# ==========================================
class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.H_att = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h = self.H_att(x_tem)
        x_v = self.V_att(x_tem)
        att = self.norm(x_h + x_v)
        out = x * self.gate_fn(att)
        return out


# ==========================================
# DCM模块：Dynamic Channel Mixing 动态通道混合
# ==========================================
class DCM(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        mid_dim = dim // reduction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, mid_dim, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_dim, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = x1 + x2
        w = self.global_pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        out = x1 * w + x2 * (1 - w)
        return out


# ==========================================
# DWNonLinear模块：Depthwise Non-linear 深度非线性增强
# ==========================================
class DWNonLinear(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.pwconv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.pwconv(self.dwconv(x)))


# ==========================================
# Linear Attention模块：线性注意力（全局）
# ==========================================
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, k_dim=16):
        super().__init__()
        self.heads = heads
        self.k_dim = k_dim
        self.to_q = nn.Conv2d(dim, k_dim * heads, 1, bias=False)
        self.to_k = nn.Conv2d(dim, k_dim * heads, 1, bias=False)
        self.to_v = nn.Conv2d(dim, k_dim * heads, 1, bias=False)
        self.to_out = nn.Conv2d(k_dim * heads, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.to_q(x).reshape(B, self.heads, self.k_dim, H * W)
        k = self.to_k(x).reshape(B, self.heads, self.k_dim, H * W)
        v = self.to_v(x).reshape(B, self.heads, self.k_dim, H * W)

        k = k.softmax(dim=-1)
        context = torch.einsum('bhkn,bhvn->bhkv', k, v)

        q = q.softmax(dim=-2)
        out = torch.einsum('bhkv,bhkn->bhvn', context, q)
        out = out.reshape(B, -1, H, W)
        out = self.to_out(out)
        return out


# ==========================================
# MultiScaleAttentionBlock_Modified：多尺度注意力模块（魔改版）
# ==========================================
class MultiScaleAttentionBlock_Modified(nn.Module):
    def __init__(self, dim, att_kernel=3, mlp_ratio=4.0, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim_split = dim // 4

        self.PA = PA(self.dim_split, norm_layer, act_layer)
        self.LA = LA(self.dim_split, norm_layer, act_layer)
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)
        self.GA = LinearAttention(self.dim_split)

        self.dcm = DCM(self.dim_split)
        self.dw_non_linear = DWNonLinear(self.dim_split * 3)  # 修改为拼接后维度

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim_split * 3, mlp_hidden_dim, 1, bias=False),  # 修正输入维度
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split] * 4, dim=1)

        x1 = self.PA(x1)
        x2 = self.LA(x2)
        x12 = self.dcm(x1, x2)

        x3 = self.MRA(x3)
        x4 = self.GA(x4)

        x_att = torch.cat((x12, x3, x4), dim=1)
        x_att = self.dw_non_linear(x_att)

        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))
        return x


# ==========================================
# 测试
# ==========================================
if __name__ == "__main__":
    x = torch.randn(2, 32, 64, 64).cuda()
    model = MultiScaleAttentionBlock_Modified(dim=32).cuda()
    out = model(x)
    print("输入shape:", x.shape)
    print("输出shape:", out.shape)
