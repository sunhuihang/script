import torch
import torch.nn as nn
from timm.models.layers import DropPath
"""
CV分割救星魔改创新2：多尺度特征融合模块（MSFF）
1. 多尺度信息捕获：在局部流和全局流的基础上，增加多个不同尺度的卷积分支，每个分支使用不同大小的
卷积核或下采样策略来提取特征。
2. 融合策略：将多尺度特征通过通道拼接或加权融合的方式进行整合，丰富特征表示。
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


# 多尺度特征融合模块
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, dim, scales=(1, 3, 5), reduction_ratio=4):
        super(MultiScaleFeatureFusion, self).__init__()
        self.dim = dim
        self.scales = scales

        self.scale_convs = nn.ModuleList([
            Conv2d_BN(dim, dim // reduction_ratio, kernel_size=scale, padding=scale // 2)
            for scale in scales
        ])
        # fuse_conv 的输入通道数为多尺度分支拼接后的通道数
        self.fuse_conv = Conv2d_BN(dim // reduction_ratio * len(scales), dim, kernel_size=1)

    def forward(self, x):
        # 分别通过不同尺度的卷积分支
        scale_features = [conv(x) for conv in self.scale_convs]
        # 通道拼接多尺度特征
        fused_features = torch.cat(scale_features, dim=1)  # 拼接后通道数为 dim // reduction_ratio * len(scales)
        # 融合后的输出
        return self.fuse_conv(fused_features)



# 修改后的 SBCFormerBlock
class SBCFormerBlock(nn.Module):  # building block
    def __init__(self, dim, resolution=7, depth_invres=2, depth_mixer=2, pool_ratio=1, scales=(1, 3, 5)):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.depth_invres = depth_invres
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        # 局部流的倒置残差块
        self.invres_blocks = nn.Sequential()
        for k in range(self.depth_invres):
            self.invres_blocks.add_module("InvRes_{0}".format(k),
                                          InvertResidualBlock(in_features=dim, hidden_features=dim * 2,
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

        # 多尺度特征融合模块
        self.msff = MultiScaleFeatureFusion(dim=dim, scales=scales)

    def forward(self, x):
        B, C, H, W = x.shape

        # 局部特征提取
        local_fea = self.invres_blocks(x)

        # 全局特征提取
        if self.pool_ratio > 1.:
            x = self.pool(x)

        # 多尺度特征融合
        fused_fea = self.msff(local_fea)

        # 恢复全局特征
        if self.pool_ratio > 1:
            fused_fea = self.convTrans(fused_fea)
            fused_fea = self.norm(fused_fea)

        # 动态融合局部与全局特征
        out = fused_fea + local_fea

        return out


if __name__ == "__main__":

    # 实例化 SBCFormerBlock
    SBCFBlock = SBCFormerBlock(dim=64, resolution=32, scales=(1, 3, 5))  # 注意： 输入特征图的分辨率resolution=H=W
    # 创建一个随机的输入张量，形状为 B C H W
    input = torch.randn(2, 64, 32, 32)
    # 将输入张量通过 SBCFormerBlock
    output = SBCFBlock(input)
    # 打印输出张量的形状
    print(f"输入形状: {input.shape}")
    print(f"输出形状: {output.shape}")
