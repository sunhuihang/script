# XMSGF: Cross-Modality Multi-Scale Guidance Fusion
# CVPR-style rework of SAFFM with cross-modality guidance, multi-scale spatial gating,
# cosine-similarity alignment, depthwise separable convs, and residual fusion.
# Author: CV缝合救星
# License: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# -------------------------
# Utils
# -------------------------
def _make_div(v: int, ratio: int, min_ch: int = 8) -> int:
    return max(v // ratio, min_ch)

class SiLU(nn.SiLU):
    """Alias for clarity."""
    pass

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: DW(kxk) + PW(1x1)."""
    def __init__(self, channels: int, kernel_size: int, stride: int = 1, padding: int = None, bias: bool = False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding,
                            groups=channels, bias=bias)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(channels)
        self.act = SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# -------------------------
# Attention blocks
# -------------------------
class ChannelGate(nn.Module):
    """CBAM-style channel attention with Avg/Max pooling -> MLP."""
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = _make_div(channels, ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        w = avg + mx
        return self.sigmoid(w)                  # [B, C, 1, 1]

class MultiScaleSpatialGate(nn.Module):
    """
    Multi-scale spatial attention:
      - For each k in {3,5,7}: DWSepConv -> channel pool (avg,max) -> 3x3 conv -> sigmoid map
      - Softmax weights over scales for adaptive aggregation.
    """
    def __init__(self, channels: int, ks: List[int] = [3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseSeparableConv(channels, k) for k in ks])
        self.to_map = nn.ModuleList([nn.Conv2d(2, 1, 3, padding=1, bias=False) for _ in ks])
        # Learnable soft selection over scales
        self.scale_logits = nn.Parameter(torch.zeros(len(ks)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = []
        for feat, head in zip(self.branches, self.to_map):
            f = feat(x)                             # [B, C, H, W]
            avg = torch.mean(f, dim=1, keepdim=True)
            mx, _ = torch.max(f, dim=1, keepdim=True)
            m = head(torch.cat([avg, mx], dim=1))   # [B, 1, H, W]
            maps.append(m)
        # soft selection across scales
        w = torch.softmax(self.scale_logits, dim=0)
        out = 0
        for wi, mi in zip(w, maps):
            out = out + wi * mi
        return self.sigmoid(out)                    # [B, 1, H, W]

def cosine_sim_map(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Channel-wise cosine similarity per spatial location:
      sim(b,1,h,w) = <x(b,:,h,w), y(b,:,h,w)> / (||x|| * ||y||)
    """
    num = (x * y).sum(dim=1, keepdim=True)
    x_norm = x.pow(2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    y_norm = y.pow(2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return num / (x_norm * y_norm)                 # [B, 1, H, W]

# -------------------------
# XMSGF Block
# -------------------------
class XMSGF(nn.Module):
    """
    Cross-Modality Multi-Scale Guidance Fusion block.

    Inputs:
      x: modality-A features (e.g., spectral/MS)   [B, C, H, W]
      y: modality-B features (e.g., spatial/PAN)   [B, C, H, W]
    Output:
      fused features                                [B, C, H, W]
    """
    def __init__(self, channels: int, ratio: int = 4, ks: List[int] = [3, 5, 7], drop: float = 0.0):
        super().__init__()
        self.chan_gate_x = ChannelGate(channels, ratio=ratio)
        self.spa_gate_y  = MultiScaleSpatialGate(channels, ks=ks)

        # Learnable mixing of guidance terms
        self.alpha = nn.Parameter(torch.tensor(1.0))   # weight for CA*SA
        self.beta  = nn.Parameter(torch.tensor(1.0))   # weight for cosine sim

        # Fusion: concat -> 1x1 (compress) -> 3x3 refine
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.norm1  = nn.GroupNorm(num_groups=1, num_channels=channels)  # LN2D-like
        self.act1   = SiLU(inplace=True)
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2  = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act2   = SiLU(inplace=True)

        self.dropout = nn.Dropout2d(p=drop) if drop > 0 else nn.Identity()

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # GroupNorm has no weight/bias by default init needed
                pass

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, "x and y must have the same shape [B,C,H,W]"
        B, C, H, W = x.shape

        # Channel attention from x; spatial attention from y
        ca = self.chan_gate_x(x)             # [B, C, 1, 1]
        sa = self.spa_gate_y(y)              # [B, 1, H, W]
        ca_sa = ca * sa                      # broadcast -> [B, C, H, W]

        # Cosine similarity alignment map (modality agreement)
        cos = cosine_sim_map(x, y).clamp(0.0, 1.0)     # [B, 1, H, W], clamp for stability
        cos = cos.expand(-1, C, -1, -1)                # match channels

        # Final guidance gate
        guide = torch.sigmoid(self.alpha * ca_sa + self.beta * cos)  # [B, C, H, W]

        # Cross-guided features
        xg = guide * x
        yg = guide * y

        z = torch.cat([xg, yg], dim=1)      # [B, 2C, H, W]
        z = self.reduce(z)
        z = self.act1(self.norm1(z))
        z = self.dropout(z)
        z = self.act2(self.norm2(self.refine(z)))

        # Lightweight residual blend (stabilizes training)
        out = z + 0.5 * (x + y)
        return out

# -------------------------
# Toy wrapper to show three-stage usage (like SAFFM1/2/3)
# -------------------------
class XMSGF3Stage(nn.Module):
    """
    Three XMSGF blocks with different scale sets,
    emulating SAFFM1/2/3 multi-scale fusion.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.b1 = XMSGF(channels, ks=[3])
        self.b2 = XMSGF(channels, ks=[5])
        self.b3 = XMSGF(channels, ks=[7])

        self.fuse12 = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.fuse123 = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f1 = self.b1(x, y)
        f2 = self.b2(x, y)
        f12 = self.fuse12(torch.cat([f1, f2], dim=1))

        f3 = self.b3(x, y)
        f123 = self.fuse123(torch.cat([f12, f3], dim=1))
        return f123

# -------------------------
# Demo / quick test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Inputs
    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = x.to(device), y.to(device)

    # Single block
    block = XMSGF(channels=C, ratio=4, ks=[3,5,7]).to(device)
    out = block(x, y)
    print(block.__class__.__name__)
    print("x:", x.shape, "y:", y.shape, "out:", out.shape)

    # Three-stage stack (optional)
    stack = XMSGF3Stage(channels=C).to(device)
    out3 = stack(x, y)
    print(stack.__class__.__name__)
    print("x:", x.shape, "y:", y.shape, "out3:", out3.shape)

    print("\n微信公众号:CV缝合救星")
