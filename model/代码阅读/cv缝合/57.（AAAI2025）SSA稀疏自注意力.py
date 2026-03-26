import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
layer_scale = True
init_value = 1e-6
'''
57. SparseViT: Nonsemantics-Centered, Parameter-Efficient Image Manipulation
Localization through Spare-Coding Transformer（AAAI 2025 顶会)
即插即用注意力模块：SSA稀疏自注意力模块
一、背景
SSA 设计初衷：稀疏自注意力（SSA）机制，旨在通过将 Vision Transformer（ViT）中的密集
全局自注意力重新设计为稀疏离散方式，让模型自适应提取非语义特征，同时大幅降低模型大小和计
算量，提高模型在 IML 任务中的泛化性和效率。

二、模块原理
1. 稀疏自注意力计算
a. 打破语义依赖：传统自注意力采用全局交互模式，在 IML 任务中会引入大量无关关键值对，且模
型过度关注语义信息，易忽略篡改后非语义信息的局部不一致。SSA 引入 “稀疏率” 超参数，将输入
特征图划分为不重叠的张量块，仅在相同颜色标记的张量块内进行自注意力计算，抑制语义信息表达，
使模型专注于提取非语义特征。
b. 降低计算复杂度：通过对特征图张量块的稀疏化处理，避免了在篡改定位中涉及大量无关关键值对
的注意力计算，减少了模型的浮点运算量（FLOPs），相比全局注意力，大约降低了 15% 的计算量，在
大规模图像处理任务中优势明显。
2. 多尺度特征融合
多尺度信息获取：在图像篡改定位任务中，不同稀疏率的特征图包含不同程度的语义和非语义信息。较小
稀疏率的特征图富含语义信息，有助于理解图像全局结构；较大稀疏率的特征图包含更多非语义信息，利
于捕捉图像细节和局部特征。在模型的 Stage 3 和 Stage 4 的不同块中引入不同稀疏率，利用不同稀
疏率块的输出作为多尺度特征图，有效获取多尺度信息。
3. 轻量级有效预测头 LFF
a. 改进特征融合方式：受 Transformer 中 Layer scale 机制启发，设计 LFF 预测头。它为每个特征
图引入可学习参数 γ 来控制缩放比例，实现更自适应的特征融合。相比传统简单的特征图相加或拼接操作，
b. LFF 能动态调整各特征图对融合结果的贡献，更好地平衡和整合多尺度特征。预测头结构与计算：LFF 
预测头由五个主要部分组成，先统一部分特征图通道维度，对部分特征图进行上采样，然后将各特征图乘以
对应的 γ 参数，再将缩放后的特征图求和并降维，最后再次上采样得到最终预测结果。通过这种设计，模型
能突出重要特征，抑制无关或冗余特征，提高检测性能。
'''

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def block(x, block_size):
    B, H, W, C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // block_size, block_size, Wp // block_size, block_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, H, Hp, C

def unblock(x, Ho):
    B, H, W, win_H, win_W, C = x.shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * win_H, W * win_W, C)
    Wp = Hp = H * win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x

def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1] % sparse_size == 0 & x.shape[2] % sparse_size == 0, 'image size should be divisible by block_size'
    grid_size = x.shape[1] // sparse_size
    out, H, Hp, C = block(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C

def alter_unsparse(x, H, Hp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp // sparse_size, Hp // sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, H)
    out = out.permute(0, 3, 1, 2)
    return out

class SSA(nn.Module):
    def __init__(self, dim, num_heads=4, sparse_size=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        self.sparse_size = sparse_size
        if self.ls:
            global init_value
            # print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        if self.ls:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x
# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module = SSA(dim=64,sparse_size=4)
    input_tensor = torch.randn(2, 64, 64, 64)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状
