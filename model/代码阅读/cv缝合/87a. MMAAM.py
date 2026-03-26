# mobile_mamba_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt

# 定义 Mish 激活函数
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 如果你没有 SS2D 实现，这里用 Identity 占位
class SS2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)

def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1', ssm_ratio=1, forward_type="v05"):
        super().__init__()
        assert in_channels == in_channels
        self.wt_levels = wt_levels
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.global_atten = SS2D(d_model=in_channels, d_state=1, ssm_ratio=ssm_ratio, initialize="v2",
                                 forward_type=forward_type, channel_first=True, k_group=2)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])# 哔哩哔哩：CV缝合救星
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', groups=in_channels * 4)
            for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)# 哔哩哔哩：CV缝合救星
        ])

    def forward(self, x):
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))
            curr_x = self.wt_function(curr_x_ll)# 哔哩哔哩：CV缝合救星
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag)).reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop() + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), x_h_in_levels.pop()], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :shapes_in_levels[i][2], :shapes_in_levels[i][3]]# B占：CV缝**合救星
        x_tag = next_x_ll
        x = self.base_scale(self.global_atten(x)) + x_tag
        return x

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# B占：CV缝**合救星
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))# B占：CV缝**合救星
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):# B占：CV缝**合救星
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM 注意力模块
class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# 动态卷积模块
class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, num_experts=4):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
            for _ in range(num_experts)
        ])
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        gates = self.gating(x)
        gates = gates.view(x.size(0), self.num_experts, 1, 1, 1)
        outputs = []
        for i, expert in enumerate(self.experts):
            outputs.append(expert(x).unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        out = (gates * outputs).sum(dim=1)
        return out

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.add_module('dwconv3x3', DynamicConv2d(in_channels, in_channels, kernel_size=kernel_size,
                                                   stride=1, padding=kernel_size // 2, groups=in_channels, bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', Mish())
        self.add_module('dwconv1x1', nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                               stride=1, padding=0, groups=in_channels, bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, groups=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))

class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = Mish()
        self.pw2 = Conv2d_BN(h, ed)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))

# 随机深度模块
class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.8):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x):
        if not self.training:
            return x
        batch_size = x.shape[0]
        random_tensor = self.survival_prob + torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x * binary_tensor / self.survival_prob

class Residual(nn.Module):
    def __init__(self, m, survival_prob=0.8):# 哔哩哔哩：CV缝合救星
        super().__init__()
        self.m = m
        self.stochastic_depth = StochasticDepth(survival_prob)

    def forward(self, x):
        return x + self.stochastic_depth(self.m(x))

class MobileMambaModule(nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25, kernels=3, ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.dim = dim
        self.global_channels = int(global_ratio * dim)
        self.local_channels = int(local_ratio * dim)
        self.identity_channels = dim - self.global_channels - self.local_channels
        self.local_op = nn.ModuleList([
            DWConv2d_BN_ReLU(self.local_channels, self.local_channels, k)
            for k in [3, 5, 7]
        ]) if self.local_channels > 0 else nn.Identity()
        self.global_op = MBWTConv2d(self.global_channels, kernel_size=kernels,
                                     ssm_ratio=ssm_ratio, forward_type=forward_type) \
            if self.global_channels > 0 else nn.Identity()
        self.cbam = CBAM(dim)
        self.proj = nn.Sequential(
            Mish(),
            Conv2d_BN(dim, dim),
            CBAM(dim)
        )

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        if isinstance(self.local_op, nn.ModuleList):
            local_features = [op(x2) for op in self.local_op]
            local_features = torch.cat(local_features, dim=1)
            local_features = torch.mean(local_features, dim=1, keepdim=True)
            local_features = local_features.expand(-1, self.local_channels, -1, -1)
        else:
            local_features = self.local_op(x2)
        out = torch.cat([self.global_op(x1), local_features, x3], dim=1)
        out = self.cbam(out)
        return self.proj(out)

class MobileMambaBlock(nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25, kernels=5, ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(dim, dim, 3, pad=1, groups=dim))
        self.ffn0 = Residual(FFN(dim, int(dim * 2)))
        self.mixer = Residual(MobileMambaModule(dim, global_ratio, local_ratio, kernels, ssm_ratio, forward_type))
        self.dw1 = Residual(Conv2d_BN(dim, dim, 3, pad=1, groups=dim))
        self.ffn1 = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))

if __name__ == '__main__':
    batch_size = 1  # 批量大小
    dim = 32  # 输入通道数
    height, width = 256, 256  # 输入图像的高度和宽度
    # 创建随机输入张量，形状为 (batch_size, dim, height, width)
    x = torch.randn(batch_size, dim, height, width)
    block = MobileMambaBlock(dim=dim)
    output = block(x)
    # 打印模型结构
    print("哔哩哔哩: CV缝合救星!")

    # 进行前向传播，得到输出
    output = block(x)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")