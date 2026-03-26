import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple

# B站：CV缝合救星
"""
CV缝合救星魔改创新：Attention-DynamicFilter
创新点说明：
1. 通道注意力增强：在频域滤波后加入通道注意力机制（ChannelAttention），使网络能自适应调整各通道特征的重要性
2. 维度优化：使用改进的通道注意力模块，支持BHWC格式输入，避免频繁的维度转换
3. 可调压缩比：新增ca_reduction参数控制注意力模块的压缩率，平衡计算量与性能
"""
class ChannelAttention(nn.Module):
    """通道注意力模块（SE Block的改进版）"""

    def __init__(self, channel, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(channel // reduction_ratio, 4)  # 确保最小维度为4

        self.mlp = nn.Sequential(
            nn.Linear(channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入形状: B, H, W, C
        x_permuted = x.permute(0, 3, 1, 2)  # 转换为B, C, H, H
        b, c, _, _ = x_permuted.shape

        y = self.avg_pool(x_permuted).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y.permute(0, 2, 3, 1)  # 恢复B, H, W, C格式


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EnhancedDynamicFilter(nn.Module):
    """改进版：在频域滤波后加入通道注意力机制"""

    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 ca_reduction=4, **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        # 主要组件
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.channel_attn = ChannelAttention(self.med_channels, ca_reduction)  # 新增通道注意力
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape

        # 动态滤波器生成
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)

        # 频域处理
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        weight = weight.view(-1, self.size, self.filter_size, self.med_channels) if self.weight_resize else weight

        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        # 新增通道注意力
        x = self.channel_attn(x)

        # 后续处理
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


if __name__ == '__main__':
    block = EnhancedDynamicFilter(32, size=64, ca_reduction=8)

    # 测试输入输出
    input = torch.rand(3, 32, 64, 64)  # 输入 B C H W
    input_bhwc = input.permute(0, 2, 3, 1)  # B H W C

    output = block(input_bhwc)
    output = output.permute(0, 3, 1, 2)  # B C H W

    print("输入形状:", input.shape)
    print("输出形状:", output.shape)  # 应保持相同形状