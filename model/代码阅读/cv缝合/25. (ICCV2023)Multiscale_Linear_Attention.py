import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from inspect import signature
from functools import partial
from typing import Dict, Tuple

"""
EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction (ICCV 2023)
即插即用模块：Multi-Scale Linear Attention(替身模块)

一、背景
传统的Transformer模型中的注意力机制在计算机视觉任务中取得了很大成功，但基于Softmax的全局注意力由于计算复杂度较高，
限制了其在高分辨率场景中的应用。而线性注意力在计算复杂度上具有优势，但在建模全局信息时存在一定的限制。为了平衡这两者
的优势，提出了多尺度线性注意力模块，通过结合线性注意力和多尺度特征表示，提升了计算效率并且保持了良好的表达能力。

二、多尺度线性注意力模块原理
1. 输入特征：与传统注意力机制一样，基于查询（Q）、键（K）、值（V）进行计算。
2. 多尺度特征：通过在不同尺度下提取特征（例如从高分辨率到低分辨率），获得不同尺度的信息，从而增强模型的表达能力。
3. 线性注意力机制：使用线性注意力机制来减少传统全局注意力计算中的O(N²)复杂度，转而使用O(Nn)的计算复杂度，其中n
为代理标记数，N为输入特征数。
4. 关键模块：
A. 尺度分离：通过多尺度特征处理，解决不同分辨率下的信息融合问题。
B. 线性聚合：在各尺度之间使用线性注意力进行信息的高效聚合。
C. 特征增强：通过深度卷积等手段恢复特征的多样性，避免信息损失。
5. 输出特征：结合多尺度信息的线性注意力模块输出增强后的特征矩阵，保留了高效的全局建模能力，并显著降低了计算复杂度。

三、适用任务
该模块适用于图像分类、目标检测、语义分割等计算机视觉任务，尤其适合处理高分辨率图像数据，在保证计算效率的同时提高
了模型的表现力。
"""

def get_same_padding(kernel_size: int or Tuple[int, ...]) -> int or Tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def build_kwargs_from_config(config: dict, target_func: callable) -> Dict[str, any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


# register activation function here
REGISTERED_ACT_DICT: Dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


# register normalization function here
REGISTERED_NORM_DICT: Dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, dilation=1, groups=1,
                 use_bias=False, dropout=0, norm="bn2d", act_func="relu", ):
        super(ConvLayer, self).__init__()
        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride), padding=padding,
                              dilation=(dilation, dilation), groups=groups, bias=use_bias, )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
            self,
            in_channels: int, out_channels: int, heads: int or None = None, heads_ratio: float = 1.0, dim=8,
            use_bias=False,
            norm=(None, "bn2d"), act_func=(None, None), kernel_func="relu", scales: Tuple[int, ...] = (5,),
            eps=1.0e-15, ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0], )
        self.aggreg = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                          bias=use_bias[0], ),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
            )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)
        self.proj = ConvLayer(total_dim * (1 + len(scales)), out_channels, 1, use_bias=use_bias[1], norm=norm[1],
                              act_func=act_func[1], )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W,), )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (qkv[..., 0: self.dim], qkv[..., self.dim: 2 * self.dim], qkv[..., 2 * self.dim:],)

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = LiteMLA(in_channels=64, out_channels=64, scales=(5,))  # scales: 单尺度:(5,); 多尺度:(3,5)
    input1 = torch.rand(3, 64, 32, 32)
    output = block(input1)
    print(input1.size())
    print(output.size())
