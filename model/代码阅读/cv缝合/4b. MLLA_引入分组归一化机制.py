import torch
import torch.nn as nn
import torch.nn.functional as F

"""
问题：MLLA 中的注意力计算依赖硬性归一化，易导致数值不稳定，尤其在处理高分辨率图像时。 

解决方案：使用分组归一化来替代全局归一化，将通道划分为若干组，以实现更稳定的注意力分布。这种方式可以减小极端值的影响，使注意力分配更加均衡。
"""
class RoPE(torch.nn.Module):
    """旋转位置编码 (Rotary Positional Embedding)"""
    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def generate_rotations(self, x):
        *channel_dims, feature_dim = x.shape[1:-1][0], x.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0, "特征维度必须能被2 * k_max整除"
        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=x.dtype, device=x.device) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d, dtype=x.dtype, device=x.device) for d in channel_dims],
                                           indexing='ij')], dim=-1)
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        return rotations

    def forward(self, x):
        rotations = self.generate_rotations(x)
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(rotations) * x_complex
        return torch.view_as_real(pe_x).flatten(-2)

class GroupedAttentionNormalization(nn.Module):
    """分组注意力归一化"""
    def __init__(self, num_groups=8, eps=1e-6):
        super(GroupedAttentionNormalization, self).__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        # 分组计算注意力归一化
        b, h, w, c = x.shape
        x = x.reshape(b, h, w, self.num_groups, c // self.num_groups)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return x.reshape(b, h, w, c)

class MLLAttentionWithGroupedNorm(nn.Module):
    """带分组归一化的 MLLA 注意力模块"""
    def __init__(self, dim=3, input_resolution=[160, 160], num_heads=4, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()
        self.group_norm = GroupedAttentionNormalization(num_groups=8)  # 分组归一化

    def forward(self, x):
        # 应用分组归一化
        x = self.group_norm(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)  # 重塑为 (B, N, C)
        n = h * w
        head_dim = c // self.num_heads

        # 计算 q 和 k
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = self.elu(qk[0]) + 1.0, self.elu(qk[1]) + 1.0
        v = x

        # 应用旋转位置编码（RoPE）
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q, k, v = [tensor.reshape(b, self.num_heads, n, head_dim) for tensor in (q, k, v)]

        # 归一化并计算注意力
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        # 将输出重塑回输入形状并添加局部位置编码（LePE）
        x = x.transpose(1, 2).reshape(b, n, c).transpose(1, 2).reshape(b, c, h, w)
        v = v.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v)  # 确保最终输出形状与输入一致
        return x

# 测试模型
if __name__ == "__main__":
    # 生成测试输入张量
    image = torch.rand(32, 64, 128, 128)
    model = MLLAttentionWithGroupedNorm(64)
    out = model(image)
    print('输入尺寸:', image.size())
    print('输出尺寸:', out.size())
