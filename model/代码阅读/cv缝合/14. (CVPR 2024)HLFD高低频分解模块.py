from einops import rearrange
import numbers
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

'''
HL-NET：使用高低频分解进行图像修复和增强 (CVPR 2024)
即插即用模块：HLDF

一、背景:
HLFD模块（High-Low Frequency Decomposition Block，高低频分解块）是HLNet方法的核心创新模块之一。
它通过将图像特征分解为高频和低频信息，并针对性地处理两种频率信息来提升图像恢复和增强的效果。HLFD模块的核心思想是：
不同类型的图像退化问题（如去噪、去模糊和超分辨率）需要不同的频率信息来达到最佳恢复效果。
1. 高频信息可以帮助恢复图像细节和纹理, 适合去噪和去模糊；
2. 低频信息则适合捕捉图像的整体结构，用于超分辨率和多帧HDR重建。

二、原理：
1. 高低频信息提取：
A. 低频通过平均池化获取特征的低频成分，捕捉图像的整体结构。
B. 高频信息提取：将原始特征减去低频信息，得到高频特征，以保留图像中的细节信息。
2. 高频和低频信息处理：
A. 高频信息处理：针对高频信息，HLFD模块使用小卷积核的局部特征提取块，通过多个小卷积核和密集连接来专注于局部细节区
域，能够更好地恢复图像的精细特征。
B. 低频信息处理：对于低频信息，HLFD模块使用全局特征提取块，通过多尺度特征融合和通道自注意力机制来捕捉长距离依赖关系
，适合恢复图像的大致轮廓和背景信息。为了防止在下采样过程中丢失结构信息，HLFD模块引入了基于小波变换的多尺度特征融合，
使模型能够有效地保留和融合结构信息。

三、适用任务：图像恢复，图像增强，图像去噪，暗光增强，超分辨率图像，目标检测，图像分割等所有CV2维任务通用涨点模块。
'''
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None
class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None
class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)
class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
class DWT_2Dfp32(nn.Module):
    def __init__(self, wave):
        super(DWT_2Dfp32, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        # self.norm = nn.LayerNorm([in_channels, 128, 128])  # Assuming input size is [224, 224]
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.gelu(x1+x)

        x2 = self.conv2(x1)
        x2 = self.gelu(x2+x1+x)

        x3 = self.conv3(x2)
        x3 = self.gelu(x3+x2+x1+x)

        x4 = self.conv4(x3)
        x4 = self.gelu(x4+x3+x2+x1+x)

        x5 = self.conv5(x4)
        x5 = self.gelu(x5+x4+x3+x2+x1+x)

        x6= self.conv6(x5)
        x6 = self.gelu(x6+x5+x4+x3+x2+x1+x)
        return x6
class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 += x  # Residual connection
        return out2
class Fusion(nn.Module):
    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        self.dwt = DWT_2D(wave)
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.high = ResNet(in_channels)
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.low = ResNet(in_channels)
        self.idwt = IDWT_2D(wave)

    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x_dwt = self.dwt(x1)
        ll, lh, hl, hh = x_dwt.split(c, 1)
        high = torch.cat([lh, hl, hh], 1)
        high1=self.convh1(high)
        high2= self.high(high1)
        highf=self.convh2(high2)
        b1, c1, h1, w1 = ll.shape
        b2, c2, h2, w2 = x2.shape

        #
        if(h1!=h2):
            x2 =F.pad(x2, (0, 0, 1, 0), "constant", 0)
        low=torch.cat([ll, x2], 1)
        low = self.convl(low)
        lowf=self.low(low)

        out = torch.cat((lowf, highf), 1)
        out_idwt = self.idwt(out)

        return out_idwt
class UNet(nn.Module):
    def __init__(self, in_channels, wave):
        super(UNet, self).__init__()
        # Define the layers
        self.trans1 = TransformerBlock(in_channels,8, 2.66, False, 'WithBias')
        self.trans2 = TransformerBlock(in_channels,8, 2.66, False, 'WithBias')
        self.trans3 = TransformerBlock(in_channels,8, 2.66, False, 'WithBias')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.upsample1 = Fusion(in_channels, wave)
        self.upsample2 = Fusion(in_channels, wave)

    def forward(self, x):
        x1=x
        # print(x1.shape)
        x1_r = self.trans1(x)
        x2 = self.avgpool1(x1)
        # print(x2.shape)
        x2_r = self.trans2(x2)
        x3 = self.avgpool2(x2)
        # print(x3.shape)
        x3_r = self.trans3(x3)
        x4 = self.upsample1(x2_r,x3_r)
        out=self.upsample2(x1_r,x4)
        b1, c1, h1, w1 = out.shape
        b2, c2, h2, w2 = x.shape

        if (h1 != h2):
            out = F.pad(out, (0, 0, 1, 0), "constant", 0)

        return out+x
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class HLFD(nn.Module):
    def __init__(self, dim, wave='haar'):
        super(HLFD, self).__init__()
        n_feats=dim
        self.down = nn.AvgPool2d(kernel_size=2)
        self.dense=Dense(n_feats)
        self.unet=UNet(n_feats, wave)

        self.alise1= nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # one_module(n_feats)

        self.att = CALayer(n_feats)

    def forward(self, x):
        low = self.down(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)
        lowf=self.unet(low)
        highfeat = self.dense(high)
        lowfeat = F.interpolate(lowf, size=x.size()[-2:], mode='bilinear', align_corners=True)
        out=self.alise2(self.att(self.alise1(torch.cat([highfeat, lowfeat], dim=1)))) + x
        return out

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 实例化模型对象
    model = HLFD(dim=32)
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 执行前向传播
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())



