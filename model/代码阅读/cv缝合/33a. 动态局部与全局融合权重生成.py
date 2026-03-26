import torch
import torch.nn as nn
import itertools
from timm.models.layers import DropPath
"""
CV缝合救星魔改创新1：动态局部与全局融合权重生成
一、不足：
在原始设计中，全局特征的激活通过固定的 h_sigmoid 函数进行调制，而局部特征和全局特征融合采用简单的通道拼接。
这种方式对局部和全局特征的依赖程度是固定的，可能在不同场景下表现出不足。
二、魔改创新：动态融合权重生成模块（Dynamic Fusion Weight Generator）：
1. 灵活性：根据局部和全局特征的特性动态调整权重，而不是简单乘法。
2. 模块设计：在融合前，通过一个轻量级的多层感知机（MLP）对局部和全局特征进行处理，生成适配的融合权重。
"""

# h_sigmoid 激活函数
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


# 标准卷积 + BN
class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# 倒置残差块
class InvertResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3, act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.pwconv1_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=1, stride=1, padding=0)
        self.dwconv_bn = Conv2d_BN(self.hidden_features, self.hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=self.hidden_features)
        self.pwconv2_bn = Conv2d_BN(self.hidden_features, self.in_features, kernel_size=1, stride=1, padding=0)

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x1 = self.pwconv1_bn(x)
        x1 = self.act(x1)
        x1 = self.dwconv_bn(x1)
        x1 = self.act(x1)
        x1 = self.pwconv2_bn(x1)

        return x + x1


# 动态融合权重生成模块
class DynamicFusionWeight(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super(DynamicFusionWeight, self).__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, local_fea, global_fea):
        # 将局部和全局特征逐通道相加并生成融合权重
        fusion = local_fea + global_fea
        weight = self.shared_mlp(fusion)
        # 用生成的权重调整局部特征
        dynamic_fea = local_fea * weight + global_fea * (1 - weight)
        return dynamic_fea


# 修改后的 SBCFormerBlock
class SBCFormerBlock(nn.Module):  # building block
    def __init__(self, dim, resolution=7, depth_invres=2, depth_mattn=1, depth_mixer=2, key_dim=16, num_heads=3, mlp_ratio=4., attn_ratio=2,
                 drop=0., attn_drop=0., drop_paths=[0.2], pool_ratio=1, invres_ratio=1, reduction_ratio=4):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.depth_invres = depth_invres
        self.depth_mattn = depth_mattn
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        # 局部流的倒置残差块
        self.invres_blocks = nn.Sequential()
        for k in range(self.depth_invres):
            self.invres_blocks.add_module("InvRes_{0}".format(k),
                                          InvertResidualBlock(in_features=dim, hidden_features=int(dim * invres_ratio),
                                                              out_features=dim, kernel_size=3, drop_path=0.))

        # 全局流的降采样与反卷积模块
        self.pool_ratio = pool_ratio
        if self.pool_ratio > 1:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            self.convTrans = nn.ConvTranspose2d(dim, dim, kernel_size=pool_ratio, stride=pool_ratio, groups=dim)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.pool = nn.Identity()
            self.convTrans = nn.Identity()
            self.norm = nn.Identity()

        # 全局特征的混合器与注意力模块
        self.mixer = nn.Sequential()
        for k in range(self.depth_mixer):
            self.mixer.add_module("Mixer_{0}".format(k),
                                  InvertResidualBlock(in_features=dim, hidden_features=dim * 2, out_features=dim,
                                                      kernel_size=3, drop_path=0.))

        # Transformer 注意力块
        self.trans_blocks = nn.Sequential()
        for k in range(self.depth_mattn):
            self.trans_blocks.add_module("MAttn_{0}".format(k),
                                         nn.Identity())  # Placeholder: 可以替换为其他模块

        # 融合模块
        self.proj = Conv2d_BN(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_fuse = Conv2d_BN(self.dim * 2, self.dim, kernel_size=1, stride=1, padding=0)
        self.dynamic_fusion = DynamicFusionWeight(dim=dim, reduction_ratio=reduction_ratio)

    def forward(self, x):
        B, C, _, _ = x.shape
        h, w = self.resolution, self.resolution
        x = self.invres_blocks(x)
        local_fea = x

        # 全局流
        if self.pool_ratio > 1.:
            x = self.pool(x)

        x = self.mixer(x)

        # 恢复全局特征
        if self.pool_ratio > 1:
            x = self.convTrans(x)
            x = self.norm(x)

        global_act = self.act(self.proj(x))

        # 动态融合局部与全局特征
        fused_fea = self.dynamic_fusion(local_fea, global_act)
        x_cat = torch.cat((x, fused_fea), dim=1)
        out = self.proj_fuse(x_cat)

        return out


if __name__ == "__main__":

    # 实例化 SBCFormerBlock
    SBCFBlock = SBCFormerBlock(dim=64, resolution=32)  # 注意： 输入特征图的分辨率resolution=H=W
    # 创建一个随机的输入张量，形状为 B C H W
    input = torch.randn(2, 64, 32, 32)
    # 将输入张量通过 SBCFormerBlock
    output = SBCFBlock(input)
    # 打印输出张量的形状
    print(f"输入形状: {input.shape}")
    print(f"输出形状: {output.shape}")
