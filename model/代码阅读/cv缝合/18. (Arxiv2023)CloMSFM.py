import torch
import torch.nn as nn
from typing import List

'''
论文题目：Rethinking Local Perception in Lightweight Vision Transformer（2023 Arxiv）
即插即用模块CloMSFM：多尺度特征融合模块，能够同时捕获高频和低频信息

一、背景
1. 视觉变换器（ViT）自提出以来，因其能够以全局的自注意力机制捕捉图像中的长距离依赖关系，迅速成为计算机视觉领域的研究热点。
ViT通过将图像划分为固定大小的patches，并将这些patches作为序列输入到变换器网络中，能够在多个视觉任务（如图像分类、
目标检测等）中取得优异的性能。
2. 然而，随着变换器模型规模的增大，计算复杂度和内存需求也大幅增加，这对于在移动设备或边缘计算设备上运行这些模型提出了很大
的挑战。尤其是在需要低延迟和低功耗的情况下，传统的视觉变换器往往无法满足实际应用的需求。

二、创新点
1. 上下文感知的局部增强
CloFormer的核心创新在于引入了上下文感知的局部增强机制。这一机制旨在解决传统视觉变换器在局部信息捕捉方面的不足。具体来
说，CloFormer通过引入局部上下文感知模块，增强了局部特征的表示能力。该模块通过为每个token（图像patch）生成专门的上下文
感知权重，能够更精确地捕捉到局部区域的语义信息。
2. 结合全局共享权重和局部上下文感知权重
传统的卷积操作通常使用全局共享权重，而ViT使用的自注意力机制则是全局的，能够捕捉到图像中的长距离依赖关系。CloFormer创新性
地结合了这两种方法——它既利用了全局共享的卷积权重，又加入了针对具体token的上下文感知权重。这样一来，CloFormer不仅能够捕捉
长距离的全局信息，还能更好地处理局部信息，从而在保持高效计算的同时，获得更强的表现。

三、适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
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

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
            # nn.Identity()
        )
    def forward(self, x):
        return self.act_block(x)
class CloMSFM(nn.Module):

    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res
    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        # 注释的这行是原论文作者的代码
        # return  self.proj_drop(self.proj(torch.cat(res, dim=1)))
        return x+ self.proj_drop(self.proj(torch.cat(res, dim=1)))


if __name__ == '__main__':
    # 实例化模型
    # dim = 64      #表示输入特征图的通道数
    # num_heads = 8 #表示多头注意力中的头数
    # group_split = [8, 8] #定义了不同的组
    # kernel_sizes = [3]  #定义了不同的组相应的卷积核大小
    # window_size = 7    # 低频注意力的窗口大小。
    #CloMSFM多尺度特点融合模块：能够同时捕获高频和低频信息
    model = CloMSFM(dim=64, num_heads=8, group_split=[4, 4], kernel_sizes=[3], window_size=7)
    # 输入一张随机图片
    input = torch.randn(1, 64, 56, 56)
    # 前向传播
    output = model(input)   #  输入 B C H W,  输出 B C H W
    print('input_size:',input.size())
    print('output_size:',output.size())
