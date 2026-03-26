import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
红外小目标检测的深度学习模型——HCF-Net (Arxiv 2024)
即插即用模块：PPA（替身模块）
一、背景：
在红外小目标检测中，主要的挑战是红外图像中的小目标通常具有较弱的热信号和不清晰的轮廓，而深度学习模型
中常见的下采样操作会导致这些小目标的信息丢失。此外，红外图像通常缺乏物理细节和对比度，容易使小目标在
复杂背景中被淹没。为了应对这一挑战，HCF-Net引入了PPA模块，旨在改进对小目标的形状和边界的描述，增强
目标定位和分割的精度。

二、PPA模块原理
1. 整体模块结构
A. 输入特征（H' × W' × C） 
B. Patch-Aware模块：分别采用不同的补丁大小参数p=2和p=4来捕捉局部和全局特征。
C. 卷积分支（Conv）：PPA模块还包含了多层卷积操作，用于捕捉更细致的特征信息。
D. 注意力模块（Attention）：融合了通道和空间注意力机制来进一步增强特征。
E. 输出特征（H' × W' × C'）：输出经过注意力模块增强后的特征。
2. Patch-Aware模块
A. Unfold 操作：首先通过 Unfold 操作将输入特征张量𝐹，拆分为多个不重叠的小补丁，补丁的大小为𝑝×𝑝，并
输出为一个具有形状 (𝑝×𝑝,𝐻′/𝑝,𝑊′/𝑝,𝐶′)的张量。
B. Mean 操作：通过在维度 -1 上进行求均值操作，得到补丁的平均值。
C. Feature Normalization (FN)：应用特征归一化来得到更加稳定的特征表示。
D. Softmax 计算：通过计算每个补丁的注意力得分（通过 Softmax），来确定每个补丁的重要性。
E. 特征选择：在计算完注意力分布后，通过选择与任务相关的特征进行进一步增强。
3. 注意力模块（Attention）：通道注意力+空间注意力。

三、适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
"""


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


# 输入 B C H W ,  输出 B C H W
if __name__ == '__main__':
    block = PPA(in_features=64, filters=64)  # in_features：输入通道数，filters：输出通道数
    input = torch.rand(3, 64, 32, 32)
    output = block(input)
    print(input.size())
    print(output.size())
