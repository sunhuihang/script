import torch
import torch.nn as nn
import torch.nn.functional as F

"""
60. CondensedAttentionNeuralBlock：用于图像恢复的高效注意力模块 CVPR（2023）
即插即用模块：CondensedAttentionNeuralBlock（CA，替身强化模块）
一、背景
图像恢复旨在从退化图像中恢复高质量图像，卷积神经网络（CNNs）和基于Transformer的方法在图像恢复中取得了一定成果，
但也存在各自的局限性。CNNs 受静态权重和稀疏连接限制，实例适应能力和全局依赖性捕捉能力不足；基于 Transformer的
方法虽能捕捉全局依赖性，但计算成本高，且注意力机制通常只能捕捉局部范围内的依赖性，无法充分发挥 Transformer的潜
力。CA旨在解决这些问题，通过新的范式高效捕捉超像素级的全局依赖性。

二、CA原理
1. 整体架构设计
CA 采用特征聚合、注意力计算和特征恢复三步范式来捕捉超像素级的全局依赖性。首先将通道和空间特征聚合为超像素特征，
然后对超像素特征依次进行通道和空间注意力计算，以充分捕捉两个维度上的全局依赖性，最后恢复特征的空间和通道维度，
使输出超像素特征的分辨率和通道数与输入像素特征一致。
2. 卷积调制块核心组件
A. 特征聚合与恢复：通过自适应方式减少通道和空间域的冗余特征获取超像素特征，并在注意力计算后恢复特征分布。具体
先沿通道维度聚合，再沿空间维度聚合，恢复时则先恢复空间特征，后恢复通道特征。
B. 通道和空间注意力：为充分捕捉超像素级全局依赖性，依次进行通道和空间注意力计算。引入通道切片和合并机制，在多
头注意力计算前后分别进行操作，以提高效率，通过特定投影、计算和合并操作完成注意力计算。
3. 微观设计考量
A. 高效捕捉全局依赖性：通过特征聚合降低维度，在可接受计算成本下进行全局注意力计算，有效捕捉超像素级全局依赖性。
B. 计算成本优势：与现有注意力机制相比，CA 在参数和计算量方面表现更优，能在保证性能的同时减少计算开销。
C. 对整体网络的贡献：为后续的双自适应神经模块（DA）提供具有丰富全局信息的超像素特征，是实现像素级全局依赖性捕捉的关键步骤。

三、适用任务：
1. CA 应用于图像恢复任务，包括灰度和彩色图像去噪、JPEG 压缩 artifact 减少、运动去模糊等。在这些任务中，CA
作为网络的关键组件，与 DA 协同工作，使网络能够有效捕捉像素级全局依赖性，提升图像恢复的性能。
2. 目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
"""
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(ChannelAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(SpatialAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = q.transpose(-2, -1) @ k * self.t

        x_ = attn.softmax(dim=-1) @ v.transpose(-2, -1)
        x_ = x_.transpose(-2, -1).contiguous()

        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class CondensedAttentionNeuralBlock(nn.Module):
    def __init__(self, embed_dim, squeezes=(4, 4), shuffle=4, expan_att_chans=4):
        super(CondensedAttentionNeuralBlock, self).__init__()
        self.embed_dim = embed_dim

        sque_ch_dim = embed_dim // squeezes[0]
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))
        sque_sp_dim = shuf_sp_dim // squeezes[1]

        self.sque_ch_dim = sque_ch_dim
        self.shuffle = shuffle
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim

        self.ch_sp_squeeze = nn.Sequential(
            nn.Conv2d(embed_dim, sque_ch_dim, 1),
            nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
        )

        self.channel_attention = ChannelAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)
        self.spatial_attention = SpatialAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)

        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x):
        x = self.ch_sp_squeeze(x)

        group_num = self.sque_ch_dim
        each_group = self.sque_sp_dim // self.sque_ch_dim
        idx = [i + j * group_num for i in range(group_num) for j in range(each_group)]
        x = x[:, idx, :, :]

        x = self.channel_attention(x)
        nidx = [i + j * each_group for i in range(each_group) for j in range(group_num)]
        x = x[:, nidx, :, :]

        x = self.spatial_attention(x)
        x = self.sp_ch_unsqueeze(x)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = CondensedAttentionNeuralBlock(32, squeezes=(4, 4), shuffle=4, expan_att_chans=4).cuda()
    input = torch.rand(3, 32, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
