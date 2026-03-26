import torch
import torch.nn as nn
import torch.nn.functional as F
'''
超级令牌视觉变换器（SViT）模块 (CVPR 2024)
即插即用模块SVit（替身模块）

一、背景
ViT的挑战与局限：视觉变换器（ViT）在众多视觉任务中取得了显著成果，但在处理高分辨率图像任务时面临计算复杂度
高的问题，因为自注意力计算成本随令牌数量呈二次增长。此外，ViT 在浅层捕获局部特征时存在高冗余现象，例如浅层
的全局注意力仅集中在少数相邻令牌上，忽略了远距离令牌，导致不必要的计算开销。虽然已有如 Swin Transformer
采用局部注意力和 Uniformer 利用卷积等方法来降低计算冗余，但这些方法在一定程度上牺牲了捕捉全局依赖的能力，
从而引发了如何在神经网络早期阶段获取高效且有效的全局表示的挑战。

超级令牌视觉变换器（SViT）是一种可应用于多种视觉任务的高效模块，通过引入超级令牌机制，解决了传统视觉变换器
在浅层捕获局部特征时的冗余问题，同时保持了对全局依赖的有效建模能力，在图像分类、对象检测、实例分割和语义分割
等任务中展现出强大性能。

二、SViT 模块原理
1. 整体架构设计
A. 输入处理与卷积茎（Conv Stem）：首先将输入图像送入由四个 3×3 卷积组成的卷积茎，其步长分别为 2、1、2、1。
这种设计能够提取更优质的局部表示，相比传统非重叠令牌化方法具有优势，并且在现代 ViT 模型中广泛应用。
B. 分层表示提取与阶段转换：经过卷积茎处理后的令牌进入四个阶段的超级令牌变换器（STT）块进行分层表示提取。在阶
段之间，使用 3×3 卷积（步长为 2）来减少令牌数量，逐步降低特征分辨率，实现层次化特征学习。
C. 预测输出阶段：最后通过 1×1 卷积投影、全局平均池化和全连接层生成最终的预测结果，完成整个视觉任务的处理流程。
2. STT块核心组件
A. 卷积位置嵌入（CPE）：采用 3×3 深度卷积来为所有令牌添加位置信息。与绝对位置编码（APE）和相对位置编码（RPE）
相比，CPE 对任意输入分辨率具有更强的适应性，能够学习到更出色的局部表示，从而为后续处理提供更准确的位置信息。
B. 超级令牌注意力（STA）机制
a. 超级令牌采样（STS）：将基于软k均值的超像素算法从像素空间迁移到令牌空间。给定视觉令牌，通过迭代计算令牌与超级
令牌之间的关联映射来采样超级令牌。具体而言，先通过平均网格区域内的令牌获取初始超级令牌，然后采用类似注意力的计算
方式迭代更新关联映射和超级令牌，且为提升效率，每个令牌仅与 9 个周围超级令牌计算关联并仅更新一次。
b. 超级令牌自注意力（MHSA）：在采样得到的超级令牌空间中应用标准自注意力机制，聚焦于全局上下文依赖关系而非局部特
征。通过特定的线性函数计算注意力分数，使模型能够在全局范围内有效聚合信息，从而更好地捕捉长距离依赖。
c. 令牌上采样（TU）：由于超级令牌在采样过程中丢失了部分局部细节，因此通过关联映射将自注意力处理后的超级令牌上采
样回原始令牌空间，并与原始令牌相加，以恢复局部信息，确保后续处理能够充分利用全局和局部特征。
3. 卷积前馈网络（ConvFFN）：由两个 1×1 卷积、一个 3×3 深度卷积和一个 GELU 非线性函数构成。该网络主要用于增强局
部表示能力，其中深度卷积有效补偿了局部相关性学习能力，与 CPE 和 STA 协同工作，使 STT 块能够同时捕获局部和长距离依
赖关系。

三、适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
'''

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)

        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)

                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)

        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


#  输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(3, 64, 64, 64).cuda()
    sa = StokenAttention(64, stoken_size=[8,8]).cuda()
    output = sa(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
