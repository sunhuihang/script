import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新：动态超级令牌大小
一、不足：
1. 计算复杂度：当前模型在stoken_forward中使用了多次 affinity_matrix的计算，尤其是在通过
迭代来更新超级令牌时，可能导致计算量过大，尤其是在输入尺寸较大的情况下。
2. 可解释性： stoken_attention中的超级令牌与像素令牌之间的关系并不容易理解，缺乏一定的可解
释性。每次计算的 affinity_matrix 更新依赖于软max操作，可能导致难以追踪其真正的贡献。
3. 缺乏适应性：超级令牌的大小是固定的，无法动态调整，且其初始化基于平均池化，这可能会影响到
模型在不同分辨率下的表现。
二、创新点：
1. 引入一种动态调整超级令牌大小的机制，基于输入图像的特点（例如局部和全局特征的大小），来自适应调
整超级令牌的大小，以提高模型的适应性和计算效率。
2. 利用输入图像的局部特征和全局特征，结合卷积网络自适应地调整超级令牌的大小。这种方法有助于在保持
全局依赖建模的同时，减少冗余计算。
"""
class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3, dim=2)

        attn = (k.transpose(-1, -2) @ q) * self.scale
        attn = attn.softmax(dim=-2)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size

        self.scale = dim ** -0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        # Introducing dynamic adjustment mechanism
        self.adjust_size_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

    def dynamic_stoken_size(self, x):
        """ Adjust the size of super tokens dynamically based on image features """
        size_map = self.adjust_size_conv(x)
        dynamic_stoken_size = torch.round(torch.sigmoid(size_map) * (self.stoken_size[0] + 2)) + 1  # Dynamically adjusts size
        return dynamic_stoken_size

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:  # 直接判断整数
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)

        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)

                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)

        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:  # 直接判断整数
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


# Test the code
if __name__ == '__main__':
    input = torch.randn(3, 64, 64, 64).cuda()
    sa = StokenAttention(64, stoken_size=[8,8]).cuda()
    output = sa(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
