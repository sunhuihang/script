import sys
import torch
import torch.nn as nn
from einops import rearrange
'''
CAFM: 卷积与注意力特征融合模块用于高光谱图像降噪 (IEEE 2024)
即插即用模块：CAFM卷积与注意力融合模块（替身模块）

一、背景
高光谱图像（HSI）因其丰富的光谱信息广泛应用于目标分类和解混等任务。然而，HSI在采集过程中经常受到多种
噪声的影响，如高斯噪声、条带噪声等，这会显著降低图像质量，影响后续分析性能。当前的HSI降噪方法要么专注
于局部特征的提取（如CNN），要么依赖于全局特征建模（如Transformer）。这些方法通常忽略了对局部和全局
特征的联合建模，限制了降噪效果。为此，本文提出了一个混合卷积与注意力网络（HCANet），其中包含卷积与注
意力融合模块（CAFM）。CAFM通过整合卷积和注意力机制，弥补二者在局部和全局建模上的不足，有效提升了降噪
性能。

二、CAFM模块原理
1. 输入特征：类似标准的自注意力机制，使用查询（Q）、键（K）和值（V）表示输入特征。
2. 卷积与注意力融合：
A. 局部分支：利用卷积操作增强局部特征建模，结合通道混洗操作，提升特征间的信息交互能力。
B. 全局分支：通过自注意力机制捕获长程依赖，全局信息通过降维操作和交互计算进一步提取。
3. 多尺度特征融合：通过多尺度前馈网络（MSFN）提取不同尺度的特征，增强非线性信息的变换能力。
4. 关键模块：
A. 局部分支：利用1×1卷积进行通道调整，结合3×3卷积和通道混洗操作，全面捕获局部特征。
B. 全局分支：采用3×3深度卷积和自注意力机制，通过计算缩减维度的注意力图，降低计算复杂度。
5. 输出融合：局部分支和全局分支的输出进行加权融合，生成最终特征。

三、适用任务：该模块适用于图像分类、目标检测、实例分割和语义分割等计算机视觉任务，尤其适用于以下场景：
1. 噪声处理：针对高斯噪声、条带噪声、脉冲噪声以及混合噪声等多种复杂噪声类型，CAFM表现出色。
2. 高效推理：通过优化计算流程和模块设计，适合资源受限的设备（如嵌入式系统）。
3. 其他视觉任务：可推广至分类、分割和检测等需要局部与全局建模的视觉任务。
'''
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output
if __name__ == '__main__':
    input = torch.rand(1, 64, 32,32)
    CAFM = CAFM(dim=64)
    output = CAFM(input)

    print('input_size:', input.size())
    print('output_size:', output.size())