import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.nn.init import trunc_normal_

# --------- utils ---------
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

# --------- LayerNorm2d with custom autograd ---------
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # d(weight) 与 d(bias)
        d_weight = (grad_output * y).sum(dim=(0, 2, 3))
        d_bias = grad_output.sum(dim=(0, 2, 3))
        return gx, d_weight, d_bias, None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# --------- Deformable Neighborhood Attention (natten-free) ---------
class DeformableNeighborhoodAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        offset_range_factor=1.0,
        stride=1,
        use_pe=True,
        dwc_pe=True,
        no_off=False,
        fixed_pe=False,
        is_causal: bool = False,
        rel_pos_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        n_head_channels = dim // num_heads

        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = num_heads
        self.nc = n_head_channels * num_heads
        self.n_groups = num_heads
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = kernel_size
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal
        if self.is_causal:
            raise NotImplementedError("is_causal=True not supported in this natten-free version.")

        kk = self.ksize
        pad_size = (kk // 2) * dilation

        # offset predictor (depthwise)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels,
                      kk, stride, pad_size, groups=self.n_group_channels, dilation=dilation),
            LayerNorm2d(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # qkv projections
        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        # 2D relative position bias (可选)
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        # depthwise conv as positional encoding
        self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)

        # 预计算 kernel 内偏移到 rpb 索引的映射（中心在中间）
        self.register_buffer("rpb_index", self._build_rpb_index(), persistent=False)

    def _build_rpb_index(self):
        k = self.ksize
        cen = k // 2
        idx = []
        # dilation 仅影响真实像素间距，不影响 rpb 的索引步长（仍是单位步）
        for dy in range(-cen, cen + 1):
            for dx in range(-cen, cen + 1):
                idx.append((dy, dx))
        # 映射到 [0..2k-2] 区间
        idx_map = torch.tensor(idx, dtype=torch.long)  # [P, 2]
        idx_map[:, 0] += (k - 1)
        idx_map[:, 1] += (k - 1)
        return idx_map  # [P, 2]

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B*g, H, W, 2
        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        k = self.ksize
        P = k * k
        pad = (k // 2) * self.dilation

        # projections
        q = self.proj_q(x)                            # [B, C, H, W]
        q_heads = q.view(B, self.n_heads, self.n_head_channels, H, W)
        q_flat = q_heads.permute(0, 1, 3, 4, 2).contiguous().view(B, self.n_heads, H*W, self.n_head_channels)  # [B,h,N,dh]
        q_flat = q_flat * self.scale

        # offsets & sampling (deformable)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # [(B*g), 2, Hg, Wg]
        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')  # [(B*g), Hg, Wg, 2]
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)  # [B*g, Hg, Wg, 2]
        if self.no_off:
            offset = offset.fill_(0.0)
        pos = offset + reference if self.offset_range_factor >= 0 else (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True
            )
            x_sampled = x_sampled.reshape(B, C, H, W)

        # relative positional encoding (depthwise)
        residual_lepe = self.rpe_table(q)

        # build local neighborhoods for K/V using unfold (supports dilation)
        k_map = self.proj_k(x_sampled)               # [B, C, H, W]
        v_map = self.proj_v(x_sampled)               # [B, C, H, W]

        k_unf = F.unfold(k_map, kernel_size=k, dilation=self.dilation, padding=pad, stride=1)  # [B, C*P, N]
        v_unf = F.unfold(v_map, kernel_size=k, dilation=self.dilation, padding=pad, stride=1)  # [B, C*P, N]

        # reshape to heads
        # K: [B, h, N, P, dh]
        k_unf = k_unf.view(B, self.n_heads, self.n_head_channels, P, H*W) \
                     .permute(0, 1, 4, 3, 2).contiguous()
        # V: [B, h, N, P, dh]
        v_unf = v_unf.view(B, self.n_heads, self.n_head_channels, P, H*W) \
                     .permute(0, 1, 4, 3, 2).contiguous()

        # attention logits: [B, h, N, P]
        # q_flat: [B,h,N,dh], k_unf: [B,h,N,P,dh]
        attn = (q_flat.unsqueeze(3) * k_unf).sum(dim=-1)  # dot(q, k)

        # add 2D relative position bias if enabled
        if self.rpb is not None:
            # rpb_index: [P, 2] with indices in [0..2k-2]
            idx = self.rpb_index  # [P, 2]
            # gather bias per head: [h, P]
            rpb_flat = self.rpb[:, idx[:, 0], idx[:, 1]]  # [h, P]
            attn = attn + rpb_flat.unsqueeze(0).unsqueeze(2)  # broadcast to [B,h,N,P]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # output: [B, h, N, dh]
        out_flat = (attn.unsqueeze(-1) * v_unf).sum(dim=3)

        # reshape back to [B, C, H, W]
        out = out_flat.view(B, self.n_heads, H, W, self.n_head_channels) \
                      .permute(0, 1, 4, 2, 3).contiguous() \
                      .view(B, C, H, W)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        y = self.proj_drop(self.proj_out(out))
        return y


# -------------------------- quick test --------------------------
if __name__ == "__main__":
    # 超参数
    batch_size = 1
    height, width = 128, 128
    channels = 64
    num_heads = 8
    kernel_size = 7
    dilation = 1

    x = torch.randn(batch_size, channels, height, width)

    attn = DeformableNeighborhoodAttention(
        dim=channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        dilation=dilation,
        offset_range_factor=1.0,
        stride=1,
        use_pe=True,
        dwc_pe=True,
        no_off=False,
        fixed_pe=False,
        is_causal=False,
        rel_pos_bias=True,   # 可开可关
        attn_drop=0.0,
        proj_drop=0.0,
    )

    with torch.no_grad():
        output = attn(x)

    print(attn)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
    # 期望: torch.Size([1, 64, 128, 128])
