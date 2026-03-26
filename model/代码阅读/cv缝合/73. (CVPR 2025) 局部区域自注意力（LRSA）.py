import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# B站：CV缝合救星
"""
73. CATANet: Efficient Content-Aware Token Aggregation for Lightweight 
    Image Super-Resolution
    中文题目：CATANet：轻量级图像超分辨率的高效内容感知标记聚合（CVPR 2025）
    所属机构：南京大学软件新技术国家重点实验室
    即插即用模块：局部区域自注意力（LRSA）模块
    
一、背景
在图像超分辨率这类低级视觉任务里，Transformer 方法虽然成果显著，但计算复杂度会随图像空间分辨率
的提升而大幅增加。当前很多方法为解决这个问题，把低分辨率图像划分成小块处理，可这样做限制了注意力
获取长距离依赖信息的能力。另外，基于聚类的方法也有缺陷，比如信息传递不够精确、推理速度慢等。为解
决这些难题，局部区域自注意力（LRSA）模块应运而生，它是 CATANet 网络的关键部分。

二、LRSA 模块介绍
（一）整体设计
LRSA 模块是 CATANet 里专门用来学习图像更精细局部细节的组件。它在 Token-Aggregation Block（TAB）
之后发挥作用，和 ConvFNN 等模块一起协同工作。LRSA 模块参考了 HPINet 的思路，采用重叠补丁的方式
来加强特征之间的交互，让模型能更精准地捕捉图像局部区域的信息，进而提升图像超分辨率的效果。
（二）核心组件与操作
1. 重叠补丁机制：LRSA 模块将输入的特征划分成相互重叠的补丁。这样一来，每个位置的感受野就扩大了，模型
可以获取到更丰富的局部上下文信息。这种方式有助于捕捉图像局部区域的细微特征和纹理，增强对局部细节的
建模能力。
2. 多头自注意力操作：针对这些重叠补丁的特征，LRSA 模块会进行多头自注意力计算。简单来说，就是把输入
的特征分别投影到查询、键和值这三个矩阵上。通过多头自注意力机制，模型能从不同的表示子空间中发现局部特
征之间的依赖关系，让对局部细节的表达能力变得更强。

三、微观设计考量
从训练优化方面来看，LRSA 模块和 CATANet 里的其他组件配合得很好。在训练时，LRSA 模块的参数会通过
反向传播不断调整优化，让它能更有效地学习局部细节特征。而且，因为它和 TAB 模块相互协作，先由 TAB 
模块捕捉长距离依赖信息，再由 LRSA 模块进一步优化局部特征，这种设计使得模型在训练过程中能更高效地利
用图像信息，提升训练效率和模型的整体性能。在实际应用中，LRSA 模块的重叠补丁机制和多头自注意力操作，
既保证了计算效率，又能精准提取局部区域的关键信息，提高图像超分辨率的质量，为后续图像重建提供更好的
特征。

四、适用任务
LRSA 模块主要用于图像超分辨率任务。在这个任务中，LRSA 模块和 CATANet 的其他组件紧密合作，让模型
既能捕捉长距离依赖信息，又能深入学习图像的局部细节。通过在多个公开的超分辨率数据集上进行实验，含有
LRSA 模块的 CATANet 在图像超分辨率任务上表现突出。和其他方法相比，它能更精准地还原图像细节，提升
重建图像的质量。像在 Urban100 等数据集上，相关指标如 PSNR 有明显的提升。
"""
def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw

def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        ouput (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output

class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q,k,v) # scaled_dot_product_attention 需要PyTorch2.0之后版本
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)
    
class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x
    
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x
     
class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, mlp_dim, heads=1):
        super().__init__()
        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        return x

if __name__ == "__main__":
    # 输入参数配置
    batch_size = 1  # Batch size
    channels = 32   # 输入通道数
    height = 256    # 高度
    width = 256     # 宽度
    ps = 16         # Patch size
    qk_dim = 36     # Query-Key维度
    mlp_dim = 96    # MLP维度
    heads = 4       # Attention头数
    
    # 创建一个输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化 LRSA 模块
    model = LRSA(dim=channels, qk_dim=qk_dim, mlp_dim=mlp_dim, heads=heads)
    print(model)
    print("哔哩哔哩: CV缝合救星!")

    # 前向传播，传入输入张量和patch大小
    output = model(x, ps)

    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

