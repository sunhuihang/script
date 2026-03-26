import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GeoPriorGen_SC(nn.Module):
    def __init__(self, embed_dim, num_heads=8, initial_value=2, heads_range=4):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H, W, depth_grid):
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H, W):
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid(index_h, index_w, indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        return mask

    def generate_1d_decay(self, l):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_structural_contrast(self, depth_map, threshold=0.05):
        B, _, H, W = depth_map.shape
        d_flat = depth_map.view(B, -1, 1)
        diff = torch.abs(d_flat - d_flat.transpose(1, 2))
        contrast_mask = (diff < threshold).float()
        return contrast_mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)

        if split_or_not:
            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            contrast_mask = self.generate_structural_contrast(depth_map)  # [B, HW, HW]
            return (sin, cos), (mask_h, mask_w), contrast_mask
        else:
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d
            contrast_mask = self.generate_structural_contrast(depth_map)
            mask = mask.unsqueeze(0) + contrast_mask.unsqueeze(1)
            return (sin, cos), mask, None

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x

def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)

class Full_GSA_SC(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()
        (sin, cos), mask, contrast_mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        vr = v.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        if split_or_not:
            # 高度方向
            qr_h = qr.permute(0, 1, 3, 2, 4)
            kr_h = kr.permute(0, 1, 3, 2, 4)
            vr_h = vr.permute(0, 1, 3, 2, 4)
            attn_h = torch.matmul(qr_h, kr_h.transpose(-1, -2)) + mask[0]
            out_h = torch.matmul(torch.softmax(attn_h, -1), vr_h)

            # 宽度方向
            qr_w = qr
            kr_w = kr
            vr_w = vr
            attn_w = torch.matmul(qr_w, kr_w.transpose(-1, -2)) + mask[1]
            out_w = torch.matmul(torch.softmax(attn_w, -1), vr_w)

            out = (out_h + out_w) / 2
        else:
            qr = qr.flatten(2, 3)
            kr = kr.flatten(2, 3)
            vr = vr.flatten(2, 3)
            attn = torch.matmul(qr, kr.transpose(-1, -2)) + mask
            out = torch.matmul(torch.softmax(attn, -1), vr)

        out = out.transpose(1, 2).reshape(bsz, h, w, -1)
        out = out + lepe
        out = self.out_proj(out)
        return out

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

# === 测试模块 ===
if __name__ == "__main__":
    embed_dim = 64
    num_heads = 8
    value_factor = 1
    batch_size = 1
    height, width = 32, 32

    input_tensor = torch.randn(batch_size, height, width, embed_dim).cuda()
    depth_map = torch.randn(batch_size, 1, height, width).cuda()

    geo_prior_gen = GeoPriorGen_SC(embed_dim=embed_dim, num_heads=num_heads).cuda()
    geo_prior = geo_prior_gen((height, width), depth_map, split_or_not=True)

    gsa_sc = Full_GSA_SC(embed_dim=embed_dim, num_heads=num_heads, value_factor=value_factor).cuda()
    output = gsa_sc(input_tensor, geo_prior, split_or_not=True)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

