import torch
import torch.nn as nn
from einops import rearrange
"""
CV缝合救星魔改创新2：增强的空间上下文建模机制
背景:
在当前模块中，局部卷积主要对小尺度局部特征进行建模，全局注意力用于捕获长程依赖。但这忽略了在中等范围内（如区域上下文）
提取空间特征，可能导致对某些中间尺度模式的感知不足。
创新:
1. 引入区域上下文增强模块（Regional Context Enhancement，简称 RCE）。
2. 在局部卷积和全局注意力之间，加入一个深度可分离卷积模块，专门提取区域上下文特征。
"""

class CAFMWithContextEnhancement(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFMWithContextEnhancement, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 原始模块
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)

        # 区域上下文增强模块 (RCE)
        self.context_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim, bias=bias),
        )

        # 局部卷积和动态融合
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)  # 添加深度维度

        # 获取 QKV 特征
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)  # 移除深度维度

        # 局部卷积分支
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # 全局注意力分支
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out_attention = (attn @ v)
        out_attention = rearrange(out_attention, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 区域上下文增强分支
        regional_context = self.context_conv(x).squeeze(2)  # 提取区域上下文

        # 融合分支输出
        out = regional_context + out_conv + out_attention
        return out


if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)
    model = CAFMWithContextEnhancement(dim=64)
    output = model(input)

    print('input_size:', input.size())
    print('output_size:', output.size())
