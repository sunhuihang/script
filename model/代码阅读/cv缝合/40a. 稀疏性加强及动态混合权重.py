import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
"""
一、不足之处：
在现有的ASSA模块中，稀疏注意力分支（SSA）和密集注意力分支（DSA）都基于标准的自注意力机制，
可能无法完全避免无关信息的干扰。具体来说，尽管引入了基于ReLU的自注意力来过滤低匹配分数的特
征，但仍然存在一定的信息损失。
二、改进点：
1. 稀疏性加强：在SSA中引入可学习的稀疏性参数，使得该模块能够自适应地调整对稀疏信息的关注度。
2. 动态混合权重：通过动态学习的加权机制来优化SSA和DSA的融合方式，而不是直接用固定的权重（w1 和 w2）。
这样可以让模型根据任务的不同动态调整两者的融合比例。
3. 引入低秩近似：在SSA模块中使用低秩近似来加速计算，并减少计算资源的消耗。
"""

# 线性投影（Linear Projection）模块，用于生成查询（Q）、键（K）和值（V）
class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        # B_表示batch大小，N表示每个输入的token数量，C表示特征维度
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)  # 扩展kv的维度
        else:
            attn_kv = x  # 如果没有给定kv，则使用x作为kv
        N_kv = attn_kv.size(1)  # kv的token数量
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)  # 生成查询
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)  # 生成键值
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


# 修改后的WindowAttention，加入了动态权重和低秩近似
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim  # 输入特征维度
        self.win_size = win_size  # 窗口大小
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的特征维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        # 相对位置偏置表，用于存储相对位置的偏置值
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))

        # 计算相对位置的索引
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化相对位置偏置

        # 生成Q、K、V的线性投影
        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("投影方式错误！")

        # 注意力dropout和输出投影
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Softmax和ReLU激活函数
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        # 动态权重，用于平衡SSA和DSA的混合
        self.dynamic_weights = nn.Parameter(torch.ones(2))  # 动态学习的权重参数，控制SSA和DSA的比例

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape  # B_是batch大小，N是token数目，C是特征维度
        q, k, v = self.qkv(x, attn_kv)  # 获取Q、K、V
        q = q * self.scale  # 对Q进行缩放
        attn = (q @ k.transpose(-2, -1))  # 计算注意力分数

        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)  # 加入相对位置偏置

        # 如果存在mask，则应用mask
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)  # 稀疏注意力
            attn1 = self.relu(attn) ** 2  # 密集注意力的低秩近似
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2

        # 使用动态权重来平衡SSA和DSA
        w1 = torch.exp(self.dynamic_weights[0]) / torch.sum(torch.exp(self.dynamic_weights))
        w2 = torch.exp(self.dynamic_weights[1]) / torch.sum(torch.exp(self.dynamic_weights))

        # 混合SSA和DSA
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        # 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        # 返回模块的额外描述
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

if __name__ == '__main__':
    # 创建一个ASSA模块实例
    model = WindowAttention_sparse(dim=64, win_size=(16, 16), num_heads=8)
    # 1,如果input是B C H W 四维的，简单代码实现如下：CV方向
    H, W = 16, 16
    input1 = torch.randn(1, 64, H, W)  #
    #将四维转三维
    input = to_3d(input1)
    output = model(input)
    output = to_4d(output,H,W)
    print('CV方向_input_size:', input1.size())
    print('CV方向_output_size:', output.size())

    # 2,如果input是B L N 三维的，简单代码实现如下：NLP方向
    # 创建一个输入张量
    input = torch.randn(1, 16 * 16, 64)  # B L N
    output = model(input)
    print('NLP方向_input_size:', input.size())
    print('NLP方向_output_size:', output.size())

