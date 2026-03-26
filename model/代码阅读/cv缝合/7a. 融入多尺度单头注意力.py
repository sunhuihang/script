import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
"""
CV缝合救星魔改创新1：融入多尺度单头注意力

背景：在多头自注意力机制中，每个注意力头可以专注于不同的特征和上下文，因此多头机制可以捕捉到更丰富的细节和不同尺度的特征。
单头自注意力的设计减少了这种多样性，可能导致模型在复杂的视觉任务中难以识别细节特征，从而影响整体表现。
这种缺点在任务需要高度细粒度特征（如细粒度图像分类或小物体检测）时可能尤其明显。

实现：将输入特征图进行多尺度变换，在不同的尺度上执行单头自注意力。执行注意力后，将不同尺度的特征图上采样回原始大小，
并与原始特征拼接或加权融合。使用一个融合层（如卷积层或逐点卷积）整合多尺度的特征，使模型既能捕获全局依赖关系，也能
提取到局部细节。


"""

# Conv2d + BatchNorm 封装
class Conv2d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.in_channels = in_channels  # 保存输入通道数，供后续使用
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

# 单头自注意力模块
class SHSA(nn.Module):
    """Single-Head Self-Attention (SHSA) 模块"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim  # 查询和键的通道数
        self.pdim = pdim  # 单头注意力的通道数

        self.pre_norm = nn.GroupNorm(1, pdim)  # 使用 GroupNorm 进行预归一化
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)  # 生成查询、键和值
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(dim, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, C - self.pdim], dim=1)  # 拆分通道，部分参与注意力
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)  # 生成查询、键和值
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)  # 根据配置划分通道
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # 调整形状以适应矩阵乘法
        attn = (q.transpose(-2, -1) @ k) * self.scale  # 计算注意力分数
        attn = attn.softmax(dim=-1)  # 归一化
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)  # 应用注意力
        return self.proj(torch.cat([x1, x2], dim=1))  # 拼接未改变的通道


# 多尺度单头自注意力模块
class MultiScaleSHSA(nn.Module):
    """
    MultiScaleSHSA: 多尺度单头自注意力模块。
    在多个尺度上执行单头注意力操作，并融合多尺度特征。
    """
    def __init__(self, dim, qk_dim, pdim, scales=(1, 0.5, 0.25)):
        super().__init__()
        self.scales = scales
        # 为每个尺度创建一个单独的SHSA模块
        self.attention_layers = nn.ModuleList([SHSA(dim, qk_dim, pdim) for _ in scales])
        # 用于融合多尺度的特征
        self.fuse = Conv2d_BN(dim * len(scales), dim, 1)

    def forward(self, x):
        multi_scale_features = []
        for scale, attn_layer in zip(self.scales, self.attention_layers):
            # 如果scale小于1，则下采样输入；否则保持原始尺寸
            scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False) if scale < 1 else x
            # 对下采样后的特征应用单头自注意力
            scaled_attn = attn_layer(scaled_x)
            # 如果进行了下采样，则将结果上采样回原始尺寸
            if scale < 1:
                scaled_attn = F.interpolate(scaled_attn, size=x.shape[2:], mode='bilinear', align_corners=False)
            multi_scale_features.append(scaled_attn)
        # 将多尺度特征拼接后通过融合层
        return self.fuse(torch.cat(multi_scale_features, dim=1))

# 用于测试的 FFN 模块
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.pw1 = Conv2d_BN(in_features, hidden_features)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(hidden_features, in_features)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))

# 用于测试的 Residual 模块
class Residual(nn.Module):
    def __init__(self, module, drop=0.):
        super().__init__()
        self.module = module
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.module(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.module(x)

# 测试完整的多尺度SHViTBlock
class SHViTBlockMultiScale(nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32, scales=(1, 0.5, 0.25)):
        super().__init__()
        self.multi_scale_attn = MultiScaleSHSA(dim, qk_dim, pdim, scales=scales)
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim))
        self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        x = self.multi_scale_attn(x)  # 多尺度自注意力
        x = self.conv(x)              # 残差卷积
        return self.ffn(x)            # 前馈网络

# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(4, 64, 32, 32)  # 创建测试输入
    shvit_block_multiscale = SHViTBlockMultiScale(dim=64)  # 实例化SHViTBlock
    output_tensor = shvit_block_multiscale(input_tensor)  # 计算输出
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
