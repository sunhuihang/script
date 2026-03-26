import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        g = self.gate(x1 + x2)
        return g * x1 + (1 - g) * x2

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class GRSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.q1, self.q2 = nn.Linear(dim//2, dim//2), nn.Linear(dim//2, dim//2)
        self.k1, self.k2 = nn.Linear(dim//2, dim//2), nn.Linear(dim//2, dim//2)
        self.v1, self.v2 = nn.Linear(dim//2, dim//2), nn.Linear(dim//2, dim//2)

        self.gate_q = GatedUnit(dim // 2)
        self.gate_k = GatedUnit(dim // 2)
        self.gate_v = GatedUnit(dim // 2)

        self.proj1, self.proj2 = nn.Linear(dim//2, dim//2), nn.Linear(dim//2, dim//2)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.channel_attn = ChannelAttention(dim)

    def forward(self, x):
        b_, n, c = x.shape
        x = x.reshape(b_, n, 2, c // 2).permute(2, 0, 1, 3)  # 2, B, N, C//2

        # gated Q/K/V
        q = self.gate_q(self.q1(x[0]), self.q2(x[1]))  # (B_, N, C//2)
        k = self.gate_k(self.k1(x[0]), self.k2(x[1]))
        v = self.gate_v(self.v1(x[0]), self.v2(x[1]))

        # 拼接为完整通道
        q = torch.cat([q, q], dim=-1)
        k = torch.cat([k, k], dim=-1)
        v = torch.cat([v, v], dim=-1)

        q = q.reshape(b_, n, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(b_, n, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(b_, n, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = x.reshape(b_, n, 2, c // 2).permute(2, 0, 1, 3)
        x = torch.stack((self.proj1(x[0]), self.proj2(x[1])), dim=0).permute(1, 2, 0, 3).reshape(b_, n, c)
        return x

class GRSAWrapper(nn.Module):
    def __init__(self, dim, window_size=(8, 8), num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., use_residual=True):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.use_residual = use_residual
        self.inner = GRSA(dim, window_size, num_heads, qkv_bias, attn_drop, proj_drop)
        self.channel_attn = ChannelAttention(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        Wh, Ww = self.window_size

        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]

        x_windows = x.view(B, C, Hp // Wh, Wh, Wp // Ww, Ww)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, Wh * Ww, C)

        x_proj = self.inner(x_windows)

        x_proj = x_proj.view(B, Hp // Wh, Wp // Ww, Wh, Ww, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        x_proj = x_proj.view(B, C, Hp, Wp)
        x_proj = self.channel_attn(x_proj)

        if pad_h > 0:
            x_proj = x_proj[:, :, :-pad_h, :]
        if pad_w > 0:
            x_proj = x_proj[:, :, :, :-pad_w]

        return x + x_proj if self.use_residual else x_proj

# 测试运行
if __name__ == "__main__":
    B, C, H, W = 1, 96, 64, 64
    x = torch.randn(B, C, H, W).cuda()
    model = GRSAWrapper(dim=C, window_size=(8, 8), num_heads=8).cuda()

    print("模型结构：")
    print(model)
    out = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
