import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新：引入跨尺度注意力机制（Cross-Scale Attention, CSA）
一、改进点：
现有的 CondensedAttentionNeuralBlock（CA）主要关注单尺度特征，而图像恢复任务往往涉及跨尺度信息的整合。
在CA的基础上引入跨尺度注意力机制（CSA），增强特征聚合的能力，提升超像素级全局依赖性的捕捉效果，同时保持计算效率。
二、核心改进：
1. 跨尺度特征提取
额外增加一个平行路径，对输入进行不同尺度的下采样（如 2×2 或 4×4 平均池化），获取多尺度特征。
将不同尺度的特征送入 CA 进行通道和空间注意力计算，确保模型能同时关注不同尺度的全局信息。
2. 跨尺度特征融合
在通道和空间注意力模块后，将不同尺度的信息进行融合。
采用 注意力加权融合 方法，让模型自主决定如何融合不同尺度的信息，而不是简单的拼接或逐点加权。
3. 计算效率优化
由于跨尺度机制会增加计算量，可以通过分组计算和低秩投影（如 1×1 卷积降维）降低计算成本。
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
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3, dim=1)
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
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3, dim=1)
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

class CrossScaleAttention(nn.Module):
    """新增：跨尺度注意力机制"""
    def __init__(self, embed_dim):
        super(CrossScaleAttention, self).__init__()
        self.pool2 = nn.AvgPool2d(2)
        self.pool4 = nn.AvgPool2d(4)
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.conv_fuse = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(self.pool2(x))
        x2 = self.conv2(self.pool4(x))
        x1 = F.interpolate(x1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)
        gate = self.gate(torch.cat([x, x1, x2], dim=1))
        x_fused = self.conv_fuse(torch.cat([x, x1, x2], dim=1)) * gate
        return x_fused

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
        self.cross_scale_attention = CrossScaleAttention(sque_sp_dim)

        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x):
        x = self.ch_sp_squeeze(x)
        x = self.cross_scale_attention(x)  # 跨尺度注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.sp_ch_unsqueeze(x)
        return x

if __name__ == '__main__':
    block = CondensedAttentionNeuralBlock(32, squeezes=(4, 4), shuffle=4, expan_att_chans=4).cuda()
    input = torch.rand(3, 32, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
