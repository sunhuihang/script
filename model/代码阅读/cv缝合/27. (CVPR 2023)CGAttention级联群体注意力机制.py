import itertools
import torch
'''
CGAttention: 级联群体注意力机制用于内存高效视觉变压器 (CVPR 2023)
即插即用模块：CGAttention 级联群体注意力机制模块（替身模块）
一、背景
Vision Transformers（ViTs）在计算机视觉任务中表现优异，但其全局自注意力机制带来的高计算和内存需求限制了其在资源
受限设备（如移动设备和嵌入式系统）中的应用。现有的一些轻量化方法虽然在减少参数和计算量上有所进展，但在实际推理速度
和性能保持方面仍存在不足。为此，本文提出了CGAttention模块，通过级联群体注意力机制，显著降低计算冗余，提高内存和计
算效率，同时保持ViTs在全局信息建模方面的优势。

二、CGAttention模块原理
1. 输入特征：与标准的多头自注意力（MHSA）机制类似，使用查询（Q）、键（K）和值（V）来表示输入特征。
2. 级联群体注意力：将注意力头进行分组，以级联的方式处理不同的特征子集，减少计算冗余并提高注意力机制的多样性。
3. 内存效率优化：通过减少MHSA与前馈网络（FFN）层之间的频繁张量变形和逐元素操作，优化内存访问效率，从而降低内存占用。
4. 关键模块：
A. 级联操作：每个注意力头接收不同的特征切片作为输入，逐步细化特征表示，增强特征的表达能力，并通过级联的方式增加模
型容量。
B. 空间操作：使用深度可分离卷积（DWC）提取空间信息，配合批量归一化和激活函数进行增强，最后通过sigmoid函数得到空间
注意力权重。
C. 通道重新分配：利用全局平均池化和卷积操作在Q、K、V投影层之间重新分配通道，突出重要通道的贡献，减少不重要组件的计
算冗余。

三、适用任务
该模块适用于图像分类、目标检测、实例分割和语义分割等计算机视觉任务，尤其适用于需要高效推理的应用场景，例如移动设备等
对内存和速度要求严格的设备。
'''
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def switch_to_deploy(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.
    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, num_heads=4,
                 attn_ratio=4,
                 resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        key_dim = dim //16
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
            attn = ((q.transpose(-2, -1) @ k) * self.scale+(trainingab[i] if self.training else self.ab[i]))
            attn = attn.softmax(dim=-1)  # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x

if __name__ == '__main__':

    input = torch.randn(1,64,32,32)
    model = CascadedGroupAttention(dim=64,resolution=32) #resolution要求和图片大小一样
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
