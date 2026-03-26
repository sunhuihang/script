import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import warnings
warnings.filterwarnings('ignore')
"""
CV缝合救星创新魔改：门控协同上下文模块：GCCM
一、CCM不足：当前的FusionAttention模块在融合局部和全局特征时，主要依赖多头自注意力机制。虽然这种机制在捕捉全局依赖关系方面表现出色，
但其计算复杂度较高，尤其是在处理高分辨率图像时计算开销和内存占用都比较大。此外，该模块在特征对齐时，只是通过相加的方式将 
local 特征与 attn 特征简单融合，没有进一步区分重要的特征信息。这样可能导致局部特征和全局特征之间的关系没有得到更充分的
表达。
二、创新魔改：为了增强局部和全局特征的融合效果，同时降低计算开销，可以引入门控机制（Gated Mechanism）来对局部和全局特征进行加权融合。
1.门控单元：我们可以在 FusionAttention 中添加一个门控单元，学习局部和全局特征的重要性，从而在融合时能够更有选择性地保留重要的特征。
2. 高效融合：门控机制可以在不显著增加计算成本的情况下，实现特征选择和加权，有助于进一步提升特征表达力。
"""
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class GatedFusion(nn.Module):
    """门控融合单元，用于局部和全局特征的加权融合"""
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        gate = self.gate(x)  # 生成门控权重
        return gate * x + (1 - gate) * y  # 根据门控权重融合x和y


class FusionAttention(nn.Module):
    def __init__(self, dim=256, ssmdims=256, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = ConvBN(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(ssmdims, dim, kernel_size=3)
        self.local2 = ConvBN(ssmdims, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding
        self.gated_fusion = GatedFusion(dim)  # 添加门控融合单元

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, y):
        B, C, H, W = x.shape

        local = self.local2(y) + self.local1(y)
        x = F.pad(x, (0, self.ws - W % self.ws, 0, self.ws - H % self.ws), mode='reflect')
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
                            h=self.num_heads, d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1) @ v
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, (0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, (0, 1, 0, 0), mode='reflect')) + local

        out = self.gated_fusion(out, local)  # 使用门控融合局部和全局特征
        out = F.pad(out, (0, 1, 0, 1), mode='reflect')
        out = self.proj(out)
        return out[:, :, :H, :W]


class CCMFusion(nn.Module):
    def __init__(self, dim=256, ssmdims=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.normx = norm_layer(dim)
        self.normy = norm_layer(ssmdims)
        self.attn = FusionAttention(dim, ssmdims, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, kernel_size=1),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, kernel_size=1),
            nn.Dropout(drop)
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.normx(x), self.normy(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    block = CCMFusion(32, 32)
    input1 = torch.rand(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)
    output = block(input1, input2)
    print('input1_size:', input1.size())
    print('input2_size:', input2.size())
    print('output_size:', output.size())
