import torch
import torch.nn as nn
from einops import rearrange
'''
Learning A Sparse Transformer Network for Effective Image Deraining （CVPR 2023）
即插即用模块：TKSA（稀疏注意力模块）（替身模块——平替VIT）
一、背景
1. 传统 Transformer 在图像去雨任务的不足凸显改进需求：在图像去雨领域，基于 Transformers 的方法
虽在利用非局部信息进行高质量图像重建方面有一定成效，但标准 Transformers 通常使用查询 - 键对的
所有相似性进行特征聚合，存在弊端。由于键的标记与查询的标记并非总是相关，在特征聚合中使用由此产生
的自注意力值会干扰潜在清晰图像的恢复，因其原生的密集计算模式会放大较小的相似性权重，使特征交互和
聚合过程易受隐式噪声影响，且在建模全局特征依赖时会考虑冗余或不相关的表示。
2. TKSA 应运而生填补关键空白：为克服上述问题，TKSA 模块被提出，旨在自适应地保留对特征聚合最有用的
自注意力值，使聚合后的特征更有利于高质量图像重建，从而提升图像去雨效果。

二、模块原理
1. 基于 top - k 选择的稀疏注意力机制设计
a. 关键注意力值筛选策略：针对标准自注意力的缺陷，设计 top - k 注意力机制。首先对输入特征进行通道
维度的编码，先应用 1×1 卷积后接 3×3 深度卷积，接着在重塑的查询和键之间计算像素对的相似性，并在转
置的注意力矩阵中屏蔽掉较低注意力权重的不必要元素。通过自适应选择前 k 个有贡献的分数，仅保留显著成
分并去除无用信息，其中 k 是可动态控制稀疏程度的参数，通过加权平均适当分数获得。在计算过程中，对注
意力矩阵每行超出范围的元素使用 scatter 函数将其概率置 0，使注意力从密集变为稀疏，避免无关信息参
与特征交互。
b. 多头部融合优化输出：采用多头策略，对每个新的查询、键和值进行多头注意力计算，将各头的输出进行拼接
后通过线性投影得到最终结果，进一步增强特征表示能力。
2. 混合尺度前馈网络协同增效
a. 多尺度信息挖掘路径：考虑到单尺度深度卷积在常规前馈网络中的局限性，MSFN 设计了两条多尺度深度卷积
路径。给定输入张量，先经层归一化后利用 1×1 卷积按比例 r 扩展通道维度，然后送入两个并行分支。在分支
中分别使用 3×3 和 5×5 深度卷积增强多尺度局部信息提取，最后将两个分支的结果融合并与输入相加，丰富了
多尺度局部信息，助力图像恢复。
b. 多尺度优势体现：这种多尺度设计能够更好地捕捉不同尺度的雨条纹信息，相较于忽视多尺度相关性的传统方
法，能显著提升图像去雨性能。

三、适用于：图像分类，目标检测、实例分割和语义分割等所有计算机视觉CV任务通用的即插即用模块
'''

class TKSA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(TKSA, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class PAM_Module(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

if __name__ == "__main__":
    input = torch.randn(5, 32, 64, 64)
    tksa = TKSA(32)
    output = tksa(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

