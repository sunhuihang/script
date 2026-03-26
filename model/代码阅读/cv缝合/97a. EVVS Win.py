import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from torchvision.transforms.functional import resize, to_pil_image

import warnings
warnings.filterwarnings('ignore')

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, w)

class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)

        x = x[:, :, :h, :w]
        return x

class PseudoMambaScan(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Conv2d(dim, dim, 1)
        self.state = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.update = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        state = self.update(x)
        return gate * x + (1 - gate) * state

class EVSS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, att=False, idx=3, patch=128):
        super().__init__()
        self.att = att
        self.idx = idx
        self.kernel_size = (patch, patch)

        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = PseudoMambaScan(dim)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

    def grids(self, x):
        b, c, h, w = x.shape
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        step_i = max(1, k1)
        step_j = max(1, k2)

        parts = []
        self.idxes = []
        for i in range(0, h - k1 + 1, step_i):
            for j in range(0, w - k2 + 1, step_j):
                parts.append(x[:, :, i:i + k1, j:j + k2])
                self.idxes.append((i, j))
        return torch.cat(parts, dim=0)

    def grids_inverse(self, outs, original_shape):
        b, c, h, w = original_shape
        preds = torch.zeros((b, c, h, w), device=outs.device)
        counts = torch.zeros((b, 1, h, w), device=outs.device)
        k1, k2 = self.kernel_size

        for idx, (i, j) in enumerate(self.idxes):
            preds[0, :, i:i + k1, j:j + k2] += outs[idx]
            counts[0, :, i:i + k1, j:j + k2] += 1

        return preds / counts

    def forward(self, x):
        if self.att:
            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1))
            if self.idx % 2 == 0:
                x = x.transpose(-2, -1)

            x_split = self.grids(x)
            x_split = x_split + self.attn(self.norm1(x_split))
            x = self.grids_inverse(x_split, x.shape)

        x = x + self.ffn(self.norm2(x))
        return x

if __name__ == '__main__':
    block = EVSS(64, att=True).to('cuda')
    input_tensor = torch.rand(1, 64, 32, 32).to('cuda')
    output = block(input_tensor)
    print('EVSS input_size:', input_tensor.size())
    print("\n哔哩哔哩：CV缝合救星\n")
    print('EVSS output_size:', output.size())
