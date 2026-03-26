import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LRSA(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
        pooled_sizes=[11,8,6,4], q_pooled_size=1, q_conv=False):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pooled_sizes]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001
        
        self.norm = nn.LayerNorm(dim)
        
        self.q_pooled_size = q_pooled_size
        
        # Useless code
        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        H, W = int(H), int(W)
        
        if self.q_pooled_size > 1:
            # Too keep the W/H ratio of the features
            q_pooled_size = (self.q_pooled_size, round(W*float(self.q_pooled_size)/H + self.eps)) \
                if W >= H else (round(H*float(self.q_pooled_size)/W + self.eps), self.q_pooled_size)
            
            # Conduct fixed pooled size pooling on q
            q = F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(B, C, H, W), q_pooled_size)
            _, _, H1, W1 = q.shape
            if self.q_conv is not None:
                q = q + self.q_conv(q)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
            else:
                q = q.view(B, C, -1).transpose(1, 2)
            q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            H1, W1 = H, W
            if self.q_conv is not None:
                x1 = x.view(B, -1, C).transpose(1, 2).reshape(B, C, H1, W1)
                q = x1 + self.q_conv(x1)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
                q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            else:
                q = self.q(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        # Conduct Pyramid Pooling on K, V
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pooled_size, l) in zip(self.pooled_sizes, d_convs):
            pooled_size = (pooled_size, round(W*pooled_size/H + self.eps)) if W >= H else (round(H*pooled_size/W + self.eps), pooled_size)
            pool = F.adaptive_avg_pool2d(x_, pooled_size)
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1))
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   # B N C
        x = x.transpose(1,2).reshape(B, -1, C)
        
        x = self.proj(x)
        
        # Bilinear upsampling for residual connection
        if self.q_pooled_size > 1:
            x = x.transpose(1, 2).reshape(B, C, H1, W1)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x.view(B, C, -1).transpose(1, 2)

        return x
    

if __name__ == "__main__":

    B = 1          # batch size
    C = 64         # embedding dim
    H, W = 32, 32  # height and width
    N = H * W

    # 初始化输入
    x = torch.randn(B, N, C)  # 输入 shape: [B, N, C]

    # 设置 pooling sizes（对应 Pyramid Pooling）
    pooled_sizes = [11, 8, 6, 4]

    # 构造深度卷积模块 d_convs（与 pooled_sizes 一一对应）, 此处参考原论文和源代码
    d_convs = nn.ModuleList([
        nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C) for _ in pooled_sizes
    ])

    # 实例
    attn = LRSA(
        dim=C,
        num_heads=4,
        qkv_bias=True,
        pooled_sizes=pooled_sizes,
        q_pooled_size=1,
        q_conv=False  # 可以设置为 True 来测试 conv 分支
    )

    out = attn(x, H, W, d_convs)

    # 打印输出形状
    print(attn)
    print("\n微信公众号:CV缝合救星\n")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
