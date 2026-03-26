import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
    def forward(self, x): return self.conv(x)

class HeatmapGuidedRingDecomposedMaSA(nn.Module):
    def __init__(self, embed_dim, num_heads=4, value_factor=1, hidden_dim=128, sigma=0.2, alpha=1.0, use_heatmap=True):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (embed_dim * self.factor) // num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor)
        self.lepe = DWConv2d(embed_dim * self.factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.center_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

        self.heatmap_predictor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.GroupNorm(4, embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
            nn.GroupNorm(4, embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, 1, 1),
            nn.Sigmoid()
        )

        self.sigma = sigma
        self.alpha = alpha
        self.use_heatmap = use_heatmap

    def forward(self, x, con_x):
        B, C, H, W = x.size()
        N = H * W
        device = x.device

        pooled = self.global_pool(x).view(B, C)
        cond_pooled = self.global_pool(con_x).view(B, C)
        pooled = torch.cat([pooled, cond_pooled], dim=1)
        dynamic_params = self.center_predictor(pooled)
        center = torch.sigmoid(dynamic_params[:, :2])
        R1 = F.softplus(dynamic_params[:, 2].unsqueeze(-1))
        R2 = F.softplus(dynamic_params[:, 3].unsqueeze(-1))

        x_perm = x.permute(0, 2, 3, 1)
        con_x_perm = con_x.permute(0, 2, 3, 1)
        q = self.q_proj(con_x_perm)
        k = self.k_proj(x_perm)
        v = self.v_proj(x_perm)

        v_lepe_input = v.permute(0, 3, 1, 2).contiguous()
        lepe = self.lepe(v_lepe_input).permute(0, 2, 3, 1)

        k = k * self.scaling

        q = q.view(B, N, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).view(-1, 2)

        masks = []
        for b in range(B):
            cur_c = center[b]
            q_d = torch.norm(coords - cur_c, dim=-1)
            k_d = q_d.clone()
            ring_q = torch.where(q_d >= R2[b], 2, torch.where(q_d >= R1[b], 1, 0))
            ring_k = torch.where(k_d >= R2[b], 2, torch.where(k_d >= R1[b], 1, 0))
            diff = torch.abs(q_d.unsqueeze(1) - k_d.unsqueeze(0))
            penalty = - (diff ** 2) / (2 * self.sigma ** 2)
            ring_penalty = self.alpha * (ring_q.unsqueeze(1) != ring_k.unsqueeze(0)).float()
            masks.append(penalty - ring_penalty)
        spatial_mask = torch.stack(masks).unsqueeze(1).expand(B, self.num_heads, N, N).clone()

        if self.use_heatmap:
            heat_input = x + con_x
            heatmap = self.heatmap_predictor(heat_input).view(B, H, W)
            heatmap = F.avg_pool2d(heatmap.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
            heatmap_flat = heatmap.view(B, N)

            for b in range(B):
                heat = heatmap_flat[b]
                dist_weight = torch.cdist(heat.unsqueeze(1), heat.unsqueeze(1))
                spatial_mask[b] = spatial_mask[b].clone() * torch.exp(-dist_weight.detach() ** 2)

            P_heat = torch.cdist(heatmap_flat.unsqueeze(2), heatmap_flat.unsqueeze(2)).clamp(min=1e-5)
            P_heat = torch.exp(-P_heat ** 2)

            gate = torch.sigmoid(torch.tanh(P_heat.unsqueeze(1) - spatial_mask))
            P_final = (1 - gate) * spatial_mask + gate * P_heat.unsqueeze(1)
        else:
            P_final = spatial_mask

        attn = torch.softmax(attn_scores + P_final, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1).view(B, H, W, -1)
        out = out + lepe
        return self.out_proj(out).permute(0, 3, 1, 2)


def main():
    import time

    # 模拟输入
    B, C, H, W = 2, 64, 32, 32  # 批大小、通道数、高度、宽度
    x = torch.randn(B, C, H, W)
    con_x = torch.randn(B, C, H, W)

    # 初始化模块（可以设置 use_heatmap=True 或 False）
    model = HeatmapGuidedRingDecomposedMaSA(embed_dim=C, use_heatmap=True)

    # 前向传播
    start_time = time.time()
    output = model(x, con_x)
    end_time = time.time()

    print(f"输出形状: {output.shape}")
    print(f"前向传播耗时: {end_time - start_time:.4f} 秒")

if __name__ == '__main__':
    main()

