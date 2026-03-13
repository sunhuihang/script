from torch.nn.utils.parametrizations import spectral_norm
import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.norm_layers import ChanLayerNorm
try:
    from .utils import apply_initialization
except:
    from utils import apply_initialization
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True,
                 spec_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if spec_norm:
            if upsampling is True:
                self.conv = nn.Sequential(*[
                    spectral_norm(nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation)),
                    nn.PixelShuffle(2)
                ])
            else:
                self.conv = spectral_norm(nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding, dilation=dilation))
        else:
            if upsampling is True:
                self.conv = nn.Sequential(*[
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
                              stride=1, padding=padding, dilation=dilation),
                    nn.PixelShuffle(2)
                ])
            else:
                self.conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding, dilation=dilation)

        self.norm = ChanLayerNorm(out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.conv)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True,
                 spec_norm=False):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace, spec_norm=spec_norm)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x):
        y = self.conv(x)
        return y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]
class ModAFNOMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 mod_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        mod_features = mod_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_features, 2 * hidden_features, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.fc1)
        apply_initialization(self.fc2)
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x, t):
        shift_msa, scale_msa = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = (shift_msa.unsqueeze(1).unsqueeze(1) + 1) * x + shift_msa.unsqueeze(1).unsqueeze(1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        return x



class ModAFNO2DLayer(nn.Module):
    def __init__(self, dim, tim_dim, fno_blocks=8, fno_softshrink=0.01, ):
        super().__init__()
        self.hidden_size = dim

        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tim_dim, 2 * dim, bias=True),
            Rearrange('b (n c) -> b n c', n=self.num_blocks)
        )
        self.softshrink = fno_softshrink
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.w1, 0)
        nn.init.constant_(self.w2, 0)
        nn.init.constant_(self.b1, 0)
        nn.init.constant_(self.b2, 0)
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    @staticmethod
    def multiply(input_data, weights):
        return torch.einsum('...bd,bdk->...bk', input_data, weights)

    def forward(self, x, t):
        B, C, H, W = x.shape
        shift_msa, scale_msa = self.adaLN_modulation(t).chunk(2, dim=-1)
        bias = x

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x.float()
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0]

        x_imag = self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1]

        x_real = x_real * (shift_msa.unsqueeze(1).unsqueeze(1) + 1) + scale_msa.unsqueeze(1).unsqueeze(1)
        x_imag = x_imag * (shift_msa.unsqueeze(1).unsqueeze(1) + 1) + scale_msa.unsqueeze(1).unsqueeze(1)
        x_real = F.relu(x_real)
        x_imag = F.relu(x_imag)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]
        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)
        return x + bias


