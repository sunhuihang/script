import torch
import torch.nn as nn
from einops import rearrange
"""
CV缝合救星魔改创新1：跨组信息融合机制
目标：解决多头注意力的组间隔离性问题，添加跨组信息融合模块，通过通道重分配和动态混合方式提升特征交互。
方法：
1. 在多头注意力输出后，引入跨组通道融合模块。
2. 使用通道重分配操作，通过全局池化和动态卷积增强组间信息共享。
3. 对不同组的特征进行混合，提高表达能力。
"""
class CAFMGroupFusion(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFMGroupFusion, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 原始注意力和卷积模块
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)

        # 跨组信息融合模块
        self.group_fusion = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // 2, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)

        # 全局分支：多头注意力
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)

        # 跨组信息融合
        fusion_weights = self.group_fusion(out)  # 动态权重
        out = out * fusion_weights  # 权重化组间融合
        out = self.project_out(out)
        out = out.squeeze(2)

        return out

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)
    model = CAFMGroupFusion(dim=64)
    output = model(input)
    print('Input size:', input.size())
    print('Output size:', output.size())
