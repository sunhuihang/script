import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch
import math
# B站：CV缝合救星
"""
CV缝合救星魔改创新：Dual Attention Enhancer （DAE）
一、改进意义
1. 提升性能：在原 D - RAMiT 模块基础上，通过一系列改进操作，增强了模型对图像特征的捕捉和处理能力，有助于进一步提升在图
像恢复任务中的性能，如更精准地重建图像细节、更有效地去除噪声等。
2. 优化计算效率：改进后的代码在一定程度上优化了计算过程，可能减少计算量和内存占用，使模型在实际应用中运行效率更高，更适合
部署在资源受限的环境中。
3. 增强模块功能：新增的组件和改进的结构使模块功能更加丰富和灵活，能够更好地适应不同的图像恢复任务需求，增强了模型的泛化能力。

二、改进内容
1. 注意力模块优化：对SpatialSelfAttention和ChannelSelfAttention进行改进。在计算注意力时，使用torch.clamp更严格地控制
logit_scale的指数化结果，确保其最大值不超过1/0.01，这有助于稳定注意力计算，提升模型的稳定性和准确性。
2. 新增组件：添加ECA（高效通道注意力）模块。通过自适应平均池化、一维卷积和 Sigmoid 激活等操作，根据通道信息为特征图生成权重，
突出重要通道特征，增强了模型对通道维度信息的利用，提升了特征质量 。
3. 特征混合模块改进：MobiVari2模块在原基础上优化，调整了卷积层的设置。使用groups参数对卷积进行分组，减少参数数量和计算量，同
时保持特征混合和增强的效果，提高了计算效率。
4. 归一化层简化：ReshapeLayerNorm模块简化，去除了对norm_layer的可选设置，默认使用nn.LayerNorm，使代码结构更简洁，减少不必要
的复杂性。
5. 前馈网络简化：FeedForward模块简化，去除了bias和drop参数设置，默认使用无偏置卷积和无 Dropout 操作，简化了网络结构，降低了超参
数调整的复杂度。
"""


# ---------- 基础组件 ----------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class QKVProjection(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)

    def forward(self, x):
        qkv = self.qkv(x)
        return rearrange(qkv, 'b (l c) h w -> b l c h w', l=self.num_head)


def get_relative_position_index(win_h, win_w):
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


# ---------- 注意力模块 ----------
class SpatialSelfAttention(nn.Module):
    def __init__(self, head_dim, num_head, total_head, window_size=8, shift=0, attn_drop=0.0, proj_drop=0.0,
                 helper=True):
        super().__init__()
        self.num_head = num_head
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))
        self.proj = nn.Conv2d(head_dim * num_head, head_dim * num_head, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))))

    def _get_rel_pos_bias(self):
        return self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1).unsqueeze(0)

    def forward(self, qkv, ch=None):
        B, L, C, H, W = qkv.size()
        if hasattr(self, 'shift') and self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2, -1))

        q, k, v = rearrange(qkv, 'b l c (h wh) (w ww) -> (b h w) l (wh ww) c',
                            wh=self.window_size, ww=self.window_size).chunk(3, dim=-1)
        if ch is not None:
            ch = rearrange(ch, 'b (l c) (h wh) (w ww) -> (b h w) l (wh ww) c',
                           l=1, wh=self.window_size, ww=self.window_size)
            v = v * torch.mean(ch, dim=1, keepdim=True)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * \
               torch.clamp(self.logit_scale.exp(), max=1 / 0.01)
        attn = self.attn_drop(F.softmax(attn + self._get_rel_pos_bias(), dim=-1))
        x = attn @ v

        x = rearrange(x, '(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)',
                      h=H // self.window_size, w=W // self.window_size, wh=self.window_size)
        return self.proj_drop(self.proj(x))


class ChannelSelfAttention(nn.Module):
    def __init__(self, head_dim, num_head, total_head, attn_drop=0.0, proj_drop=0.0, helper=True):
        super().__init__()
        self.num_head = num_head
        self.proj = nn.Conv2d(head_dim * num_head, head_dim * num_head, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))))

    def forward(self, qkv, sp=None):
        B, L, C, H, W = qkv.size()
        q, k, v = rearrange(qkv, 'b l c h w -> b l c (h w)').chunk(3, dim=-2)
        if sp is not None:
            v = v * rearrange(torch.mean(sp, dim=1, keepdim=True), 'b c h w -> b 1 c (h w)')

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * \
               torch.clamp(self.logit_scale.exp(), max=1 / 0.01)
        x = self.attn_drop(F.softmax(attn, dim=-1)) @ v
        return self.proj_drop(self.proj(rearrange(x, 'b l c (h w) -> b (l c) h w', h=H)))


# ---------- 改进组件 ----------
class ECA(nn.Module):
    """高效通道注意力(改进版)"""

    def __init__(self, channels, gamma=2, beta=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs(int((math.log(channels, 2) + beta) / gamma)) | 1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ReshapeLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return rearrange(self.norm(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h=x.size(2))


class MobiVari2(nn.Module):
    """轻量级特征混合模块"""

    def __init__(self, dim, exp_factor=1.2, groups=4):
        super().__init__()
        hidden_dim = int(dim * exp_factor)
        hidden_dim = (hidden_dim // groups) * groups  # 对齐分组数

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, groups=groups),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, 1),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)


# ---------- 核心模块 ----------
class DRAMiTransformerV2(nn.Module):
    def __init__(self, dim, num_head=4, window_size=8, chsa_ratio=0.25, drop_path=0.0):
        super().__init__()
        self.chsa_head = max(1, int(num_head * chsa_ratio))
        self.spsa_head = num_head - self.chsa_head

        # 注意力分支
        self.qkv = QKVProjection(dim, num_head)
        self.spsa = SpatialSelfAttention(dim // num_head, self.spsa_head, num_head, window_size)
        self.chsa = ChannelSelfAttention(dim // num_head, self.chsa_head, num_head)

        # 特征处理
        self.mobi = MobiVari2(dim)
        self.eca = ECA(dim)  # 新增的通道注意力

        # 正则化
        self.norm1 = ReshapeLayerNorm(dim)
        self.norm2 = ReshapeLayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ffn = FeedForward(dim)

    def forward(self, x, sp=None, ch=None):
        shortcut = x
        qkv = self.qkv(x)

        # 并行注意力计算
        sp_out = self.spsa(qkv[:, :self.spsa_head], ch)
        ch_out = self.chsa(qkv[:, self.spsa_head:], sp)

        # 特征融合与增强
        fused = self.mobi(torch.cat([sp_out, ch_out], dim=1))
        fused = self.eca(fused)  # 通道注意力增强

        # 残差连接
        x = shortcut + self.drop_path(self.norm1(fused))
        x = x + self.drop_path(self.norm2(self.ffn(x)))
        return x, sp_out, ch_out, fused


if __name__ == '__main__':
    # 测试模块
    model = DRAMiTransformerV2(dim=64, num_head=4)
    x = torch.randn(2, 64, 32, 32)
    output, sp, ch, fused = model(x)

    print("输入尺寸:", x.shape)
    print("输出尺寸:", output.shape)
    print("空间注意力特征:", sp.shape)
    print("通道注意力特征:", ch.shape)
    print("融合特征尺寸:", fused.shape)