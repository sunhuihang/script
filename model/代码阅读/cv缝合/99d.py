import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from typing import List
import antialiased_cnns
from torch import Tensor


# 更新版的 PA（点注意力模块）
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
        x = x * self.gate_fn(att)  # 通过门控函数调节注意力
        return x


# 更新版的 MRA（多尺度注意力模块）
class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)

        # 引入动态卷积核的方式，使得网络能根据输入数据的特征大小自动调整计算区域
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_kernel // 2, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_kernel // 2), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_kernel // 2, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_kernel // 2), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)

        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)

        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        # 动态加权注意力输出，使得每个特征的加权更加灵活
        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att), size=(x.shape[-2], x.shape[-1]), mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# 实现 LA（局部注意力模块）
class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        return self.conv(x)


# 实现 D_GA（全局注意力模块）
class D_GA(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten the height and width dimensions into a sequence
        x = x.view(B, C, -1).permute(2, 0, 1)  # [seq_len, batch_size, dim]
        attn_output, _ = self.attn(x, x, x)
        x = self.proj(attn_output)
        # Restore the shape [batch_size, channels, height, width]
        x = x.permute(1, 2, 0).view(B, C, H, W)
        x = self.norm(x)
        return x


# 实现 GA（全局注意力）
class GA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten the height and width dimensions into a sequence
        x = x.view(B, C, -1).permute(2, 0, 1)  # [seq_len, batch_size, dim]
        attn_output, _ = self.attn(x, x, x)
        x = self.proj(attn_output)
        # Restore the shape [batch_size, channels, height, width]
        x = x.permute(1, 2, 0).view(B, C, H, W)
        x = self.norm(x)
        return x


# 实现 GA12（跨阶段全局注意力）
class GA12(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten the height and width dimensions into a sequence
        x = x.view(B, C, -1).permute(2, 0, 1)  # [seq_len, batch_size, dim]
        attn_output, _ = self.attn(x, x, x)
        x = self.proj(attn_output)
        # Restore the shape [batch_size, channels, height, width]
        x = x.permute(1, 2, 0).view(B, C, H, W)
        x = self.norm(x)
        return x


# 更新版的 LWGA_Block（加深网络模型的融合性与稳定性）
class LWGA_Block(nn.Module):
    def __init__(self, dim, stage, att_kernel, mlp_ratio, drop_path, act_layer, norm_layer):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)

        self.PA = PA(self.dim_split, norm_layer, act_layer)  # 点注意力模块
        self.LA = LA(self.dim_split, norm_layer, act_layer)  # 局部注意力模块
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # 多尺度注意力模块

        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)  # 全局注意力模块
        elif stage == 3:
            self.GA4 = GA(self.dim_split)  # 全局注意力模块（阶段3）
            self.norm = norm_layer(self.dim_split)
        else:
            self.GA12 = GA12(self.dim_split, act_layer)  # 跨阶段全局注意力
            self.norm = norm_layer(self.dim_split)

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)

        # 增强不同注意力机制的组合与传递
        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)

        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))

        # 特征融合与下游计算
        x_att = torch.cat((x1, x2, x3, x4), 1)
        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))
        return x


# 示例代码：初始化并前向传播
if __name__ == "__main__":

    batch_size = 1
    dim = 32
    height, width = 20, 20
    input_tensor = torch.randn(batch_size, dim, height, width).cuda()

    # 初始化 LWGA_Block
    model = LWGA_Block(
        dim=dim,
        stage=2,
        att_kernel=3,
        mlp_ratio=4.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d
    ).cuda()

    print(model)
    print("\n 哔哩哔哩：CV缝合救星!\n")

    # 前向传播
    output = model(input_tensor)

    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
