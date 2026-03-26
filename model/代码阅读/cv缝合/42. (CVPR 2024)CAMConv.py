import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from basicsr.archs.arch_util import flow_warp
import warnings
warnings.filterwarnings('ignore')
'''
用于大图像超分辨率的内容感知混合器（CAMixer）：提升超分辨率任务的性能与计算平衡（CVPR 2024）
即插即用模块：CAMixer（替身添花模块）

一、背景：随着对大图像（2K - 8K）超分辨率（SR）需求的增长，现有方法存在缺陷。加速框架基于内容感知路由，
但存在分类和分区不佳、感受野有限的问题；轻量级模型设计虽能减少计算，但未考虑内容的判别处理，且在大输入图
像上的应用未被充分研究。二者独立，缺乏协作。

二、CAMixer 模块原理
1. 输入特征处理：输入图像特征，先通过一个点卷积对其进行投影操作，得到处理后的特征，为后续计算做准备。
2. 模块组成与处理
A. 预测器：综合输入特征相关的局部信息、全局信息和位置编码信息，计算得到一个共享的中间特征图。接着基于这个
中间特征图生成多个重要元素，包括用于调整窗口形状的偏移信息、决定窗口计算方式的掩码以及用于增强卷积分支的
空间和通道注意力。这些元素能够根据图像内容动态地引导后续计算。
B. 自注意力分支：利用预测器生成的偏移信息，通过双线性插值调整输入特征，使窗口能够包含更多有用内容。然后根
据窗口形状重新排列特征，在训练和推理阶段分别通过特定方式确定哪些窗口进行自注意力计算，哪些进行卷积计算。最
后计算复杂窗口的自注意力，以便更好地处理图像中的复杂区域。
C. 卷积分支：利用深度卷积和预生成的通道注意力捕获局部相关性，通过简单的乘法操作实现对特征的简单注意力处理，
然后整合自注意力分支和卷积分支的输出，再经过一个点卷积投影得到最终输出，确保处理后的特征能有效用于后续超分
辨率任务。
3. 输出：经过上述一系列操作，CAMixer 输出处理后的特征，这些特征将用于后续的超分辨率操作，为生成高质量的超
分辨率图像提供有力支持。

三、适用于：超分辨率图像任务，小目标检测，小目标分割，图像增强，暗光增强等所有CV2维任务

'''
class CAMConv (nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.project_q = nn.Linear(dim, dim, bias=bias)
        self.project_k = nn.Linear(dim, dim, bias=bias)

        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k // 2) * d), groups=dim, dilation=d))
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)

        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim, window_size, ratio=ratio)
        # self.route = PredictorLG(dim, window_size, ratio=ratio)

    def forward(self, x, condition_global=None, mask=None, train_mode=False):
        N, C, H, W = x.shape

        v = self.project_v(x)

        if self.is_deformable:
            condition_wind = torch.stack(
                torch.meshgrid(torch.linspace(-1, 1, self.window_size), torch.linspace(-1, 1, self.window_size))) \
                .type_as(x).unsqueeze(0).repeat(N, 1, H // self.window_size, W // self.window_size)
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        mask, offsets, ca, sa = self.route(_condition, ratio=self.ratio, train_mode=train_mode)

        q = x
        k = x + flow_warp(x, offsets.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border')
        qk = torch.cat([q, k], dim=1)

        vs = v * sa

        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        if self.training or train_mode:
            N_ = v.shape[1]
            v1, v2 = v * mask, vs * (1 - mask)
            qk1 = qk * mask
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1, v2 = batch_index_select(v, idx1), batch_index_select(vs, idx2)
            qk1 = batch_index_select(qk, idx1)

        v1 = rearrange(v1, 'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1, 'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn @ v1

        f_attn = rearrange(f_attn, '(b n) (dh dw) c -> b n (dh dw c)',
                           b=N, n=N_, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H // self.window_size, w=W // self.window_size, dh=self.window_size, dw=self.window_size
        )

        out = attn_out
        out = self.act(self.conv_sptial(out)) * ca + out
        out = self.project_out(out)
        # if self.training:
        #     return out, torch.mean(mask, dim=1)
        return out

    def calculate_flops(self, x):
        H, W = x.shape[2:]
        flops = np.longlong(0)
        # predictor
        cdim = self.dim + 4
        flops += H * W * cdim * cdim // 4 * 2
        flops += H * W * cdim // 4 * (2 + 1) * 2
        flops += H * W * cdim // 4 * 2 * 2
        flops += math.ceil(H / self.window_size) * math.ceil(
            W / self.window_size) * self.window_size ** 2 * self.window_size * 2
        flops += math.ceil(H / self.window_size) * math.ceil(
            W / self.window_size) * self.window_size * 2 * self.window_size * 2
        # attn
        flops += H * W * self.dim * self.dim * 2
        flops += 2 * H * W * self.dim * self.dim * 2 * self.ratio
        Hp = self.window_size * math.ceil(H / self.window_size)
        Wp = self.window_size * math.ceil(W / self.window_size)
        Np = Hp * Wp
        nW = int(Np * self.ratio / self.window_size / self.window_size)
        window_flops = 2 * (self.window_size * self.window_size) ** 2 * self.dim
        flops += nW * window_flops
        # conv
        flops += 2 * H * W * self.dim * 9 * 2
        flops += H * W * self.dim * self.dim * 2
        return flops
def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """

    def __init__(self, dim, window_size=8, k=2, ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size ** 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim // 4, 1),
            LayerNorm(cdim // 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim // 4, cdim // 8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim // 8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim // 4, dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):

        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)

        x = torch.mean(x, keepdim=True, dim=1)

        x = rearrange(x, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training or train_mode:
            return mask, offsets, ca, sa
        else:
            score = pred_score[:, :, 0]
            B, N = score.shape
            r = torch.mean(mask, dim=(0, 1)) * 1.0
            if self.ratio == 1:
                num_keep_node = N  # int(N * r) #int(N * r)
            else:
                num_keep_node = min(int(N * r * 2 * self.ratio), N)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], offsets, ca, sa

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 实例化模型对象
    model = CAMConv (dim=32)
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 执行前向传播
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())