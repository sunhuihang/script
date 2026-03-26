import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# CV缝合救星2025.06.08视频
# 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
# 门控融合模块
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fuse = torch.cat([x1, x2], dim=1)
        alpha = self.gate(fuse)
        return alpha * x1 + (1 - alpha) * x2

# 魔改 ACFM 模块
class ACFMAttentionPlus(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.scale = (dim // num_heads) ** -0.5

        # QKV 提取与深度卷积处理
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        # 局部路径残差卷积
        self.local_fc = nn.Conv3d(3 * num_heads, 9, kernel_size=1)
        self.dep_conv = nn.Conv3d(9 * dim // num_heads, dim, kernel_size=3, padding=1, groups=dim // num_heads)

        # 多尺度并行注意力（下采样路径）
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 相对位置偏置（可学习）
        self.relative_bias = nn.Parameter(torch.randn(num_heads, 1, 1))  # 简化形式

        # 门控融合
        self.fusion = GatedFusion(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x3d = x.unsqueeze(2)  # (B, C, 1, H, W)

        # QKV 提取
        qkv = self.qkv_dwconv(self.qkv(x3d))  # (B, 3C, 1, H, W)
        qkv = qkv.squeeze(2)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # 多尺度下采样路径构建第二组注意力
        x_down = self.downsample(x)  # (B, C, H//2, W//2)
        qkv_down = self.qkv_dwconv(self.qkv(x_down.unsqueeze(2))).squeeze(2)
        q_d, k_d, v_d = qkv_down.chunk(3, dim=1)

        # reshape for attention
        def reshape_qkv(tensor):
            return rearrange(tensor, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q1 = F.normalize(reshape_qkv(q), dim=-1)
        k1 = F.normalize(reshape_qkv(k), dim=-1)
        v1 = reshape_qkv(v)

        q2 = F.normalize(reshape_qkv(q_d), dim=-1)
        k2 = F.normalize(reshape_qkv(k_d), dim=-1)
        v2 = reshape_qkv(v_d)

        # attention path 1
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature + self.relative_bias
        attn1 = attn1.softmax(dim=-1)
        out1 = attn1 @ v1
        out1 = rearrange(out1, 'b h c (h1 w1) -> b (h c) h1 w1', h1=H, w1=W)

        # attention path 2 (multi-scale)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature + self.relative_bias
        attn2 = attn2.softmax(dim=-1)
        out2 = attn2 @ v2
        out2 = rearrange(out2, 'b h c (h1 w1) -> b (h c) h1 w1', h1=H//2, w1=W//2)
        out2 = self.upsample(out2)

        # Attention融合
        attn_out = out1 + out2
        attn_out = self.project_out(attn_out.unsqueeze(2)).squeeze(2)

        # Local conv 分支
        f_all = qkv.reshape(B, H*W, 3*self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.local_fc(f_all.unsqueeze(2)).squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(B, 9*C//self.num_heads, H, W)
        out_conv = self.dep_conv(f_conv.unsqueeze(2)).squeeze(2)

        # Gated fusion
        out = self.fusion(attn_out, out_conv)
        return out

# 测试代码
if __name__ == "__main__":
    x = torch.randn(1, 32, 128, 128).cuda()
    model = ACFMAttentionPlus(dim=32, num_heads=4).cuda()
    y = model(x)
    print(model)
    print("哔哩哔哩:CV缝合救星")
    print("输入：", x.shape)
    print("输出：", y.shape)
