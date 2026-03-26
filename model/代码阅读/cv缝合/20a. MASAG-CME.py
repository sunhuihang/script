import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改：MASAG-CME，其中 CME 代表Channel Mix Enhancer。
不足：
1. 缺乏特征通道的细致处理：原有模型对融合后的通道权重未进行有效的动态调整。
2. 没有对融合特征的增强机制：融合后的特征缺少进一步的增强手段来提升模型的区分能力。
创新魔改：通道混合增强模块（CMEM）：
通过引入CMEM模块，模型可以根据学习到的缩放因子动态增强重要特征。
"""
class GlobalExtraction(nn.Module):
    def __init__(self, dim):
        super(GlobalExtraction, self).__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1),
            nn.BatchNorm2d(1)
        )

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim=1, keepdim=True)[0]
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)
        x2 = self.maxpool(x_)
        cat = torch.cat((x, x2), dim=1)
        proj = self.proj(cat)
        return proj

class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction=None):
        super(ContextExtraction, self).__init__()
        self.reduction = 1 if reduction is None else 2

        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv2dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv2d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      padding=1,
                      groups=dim),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      padding=2,
                      dilation=2),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            nn.Conv2d(in_channels=dim,
                      out_channels=dim // self.reduction,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=dim // self.reduction)
        )
        return proj

    def forward(self, x):
        x = self.dconv(x)
        x = self.proj(x)
        return x

class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super(MultiscaleFusion, self).__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction(dim)
        self.bn = nn.BatchNorm2d(num_features=dim)

    def forward(self, x, g):
        x = self.local(x)
        g = self.global_(g)
        fuse = self.bn(x + g)
        return fuse

class ChannelMixEnhancement(nn.Module):
    """新的通道混合增强模块 (CMEM)，通过混合不同通道特征来增强整体表达"""
    def __init__(self, dim, reduction_ratio=4):
        super(ChannelMixEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MASAG(nn.Module):
    def __init__(self, dim):
        super(MASAG, self).__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1)
        )
        self.cmem = ChannelMixEnhancement(dim)  # 新增通道混合增强模块

    def forward(self, x, g):
        x_ = x.clone()
        g_ = g.clone()
        multi = self.multi(x, g)  # B, C, H, W
        multi = self.selection(multi)  # B, num_path, H, W

        attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
        A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

        x_att = A.expand_as(x_) * x_
        g_att = B.expand_as(g_) * g_
        x_att = x_att + x_
        g_att = g_att + g_

        # 双向交互
        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att
        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att
        interaction = x_att_2 * g_att_2

        # 使用通道混合增强模块
        enhanced = self.cmem(interaction)

        projected = torch.sigmoid(self.bn(self.proj(enhanced)))
        weighted = projected * x_
        y = self.conv_block(weighted)
        y = self.bn_2(y)
        return y

if __name__ == "__main__":
    # 创建输入特征图
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)

    # 创建MASAG实例
    masag = MASAG(dim=64)

    # 将两个输入特征图传递给MASAG模块
    output = masag(input1, input2)

    # 打印输入和输出的尺寸
    print(f"input 1 shape: {input1.shape}")
    print(f"input 2 shape: {input2.shape}")
    print(f"output shape: {output.shape}")
