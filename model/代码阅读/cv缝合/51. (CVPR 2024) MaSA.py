import torch
import torch.nn as nn
from typing import Tuple, Union

from ultralytics.nn.modules import C3

'''
RMT:Retentive Networks Meet Vision Transformers（CVPR 2024 顶会论文）
Manhattan Self - Attention (MaSA) 模块（替身添花模块）
一、背景
1. ViT 局限引发创新需求：Vision Transformer（ViT）的 Self-Attention 模块存在不足，缺乏明确
空间先验且计算复杂度高，在图像任务中应用受限。例如在图像识别时难以有效利用空间信息，计算成本高
昂影响效率。
2. MaSA 旨在解决问题：受 NLP 领域 Retentive Network（RetNet）启发，MaSA 应运而生。其目标是
将 RetNet 的时间衰减机制拓展到空间域，为视觉骨干网络引入基于曼哈顿距离的空间先验，并通过特定注意
力分解形式降低全局信息建模计算负担，提升视觉任务性能。

二、模块原理
1. 从 RetNet 到 MaSA 的转变
a. 单向到双向衰减的拓展：RetNet 因文本数据因果性，保留机制为单向。MaSA 为适配图像任务将其变为双向，
即让每个 token 能关注前后的 token，克服了单向的局限，使模型能更好地处理图像信息。
b. 一维到二维衰减的延伸：进一步把一维保留机制扩展到二维图像空间。依据图像 token 的二维坐标，基于曼
哈顿距离重新定义空间衰减矩阵，距离远的 token 间注意力分数衰减大，为模型提供丰富空间先验。同时，MaSA
采用 Softmax 引入非线性，避免了 RetNet 门控函数带来的额外问题。
2. 分解的 Manhattan Self - Attention
a. 创新分解方法：针对早期视觉骨干中 token 多导致计算成本高的问题，MaSA 沿图像轴分解。分别计算水平和
垂直方向注意力分数，再组合结果，在分解 Self - Attention 同时分解空间衰减矩阵，以线性复杂度建模全局
信息且保持空间先验，接收域形状不变。
b. 局部增强策略：引入基于深度可分离卷积（DWConv）的局部上下文增强模块（LCE），进一步提升 MaSA 的局
部表达能力，增强模型性能。

三、适用于：图像分类，目标检测、实例分割和语义分割等所有计算机视觉CV任务通用的即插即用模块
'''

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x
class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads=4, initial_value=1, heads_range=3):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=True):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class MaSA(nn.Module):

    def __init__(self, embed_dim, num_heads=4, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.RetNetRelPos2d = RetNetRelPos2d(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        x = x.permute(0,2,3,1)
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = self.RetNetRelPos2d((h,w))

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)  # (b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # (b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # (b h n w w)
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # (b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # (b w n h h)
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)

        return output.permute(0,3,1,2)

# 输入 B H W C , 输出 B  H W C
if __name__ == "__main__":
    module = MaSA(64)  # 创建 MaSA模块实例，输入通道数为 64
    input_tensor = torch.randn(1,64,128, 128)  # 创建一个形状为 (1, 64,128, 128) 的随机输入张量
    output_tensor = module(input_tensor)  # 通过MaSA模块计算输出
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状