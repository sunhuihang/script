import torch
import torch.nn as nn
import torch.nn.functional as F
'''
SMFANet：轻量级自调制特征聚合网络,用来实现高效的图像超分辨率   (ECCV 2024)
即插即用特征融合模块：SMFA
一、背景：传统的超分辨率技术面临计算量大、内存消耗高的问题，尤其是Transformer架构中的自注意力（Self-Attention）
机制虽然能捕捉非局部信息，却存在计算开销过大的缺陷。自注意力机制还表现出低频偏好性（low-pass nature），在细节重
建方面能力不足，容易产生模糊效果。

二、SMFA模块
1. SMFA模块由两部分组成：一种高效的自注意力近似（EASA）分支用于非局部信息建模，以及一种局部细节估计（LDE）分支用
于捕捉局部细节。
2. EASA分支通过下采样和方差调制来捕捉全局特征，减少了传统自注意力的计算需求。
3. LDE分支利用卷积层进行局部特征提取，使得模型可以兼顾非局部结构信息和局部细节，达到更精确的重建效果。

三、适用任务：高分辨率图像重建，暗光增强，图像恢复，等所有CV任务上通用特征融合模块
'''
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x
class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1,32,256,256)
    model = SMFA(dim=32)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
