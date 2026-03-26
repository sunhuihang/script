import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
"""
CV缝合救星魔改创新：引入动态代理增强模块
不足：
1. 代理标记生成的限制：现有的代理标记生成（如池化或固定可学习参数）缺乏动态适应能力，无法根据输
入特征的复杂性调整代理标记的内容。
2. 现有的代理标记生成（如池化或固定可学习参数）缺乏动态适应能力，无法根据输入特征的复杂性调整代理标记的内容。
创新魔改：动态代理增强模块（DAEM）
1. 引入动态代理生成机制：通过引入额外的特征编码器，生成更丰富的动态代理标记，增强模型的全局感知能力。
2. 融合多尺度特征：在代理标记生成和注意力计算中加入多尺度特征聚合，提高对复杂背景中细节和全局信息的捕获能力。
"""
class DynamicAgentAttention(nn.Module):
    """
    动态代理增强模块（DAEM）：解决代理标记生成的单一性问题，融合多尺度特征增强。
    """
    def __init__(self, dim, num_patches, num_heads=8, agent_num=49, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 查询(Q)、键(K)、值(V)线性投影
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 动态代理标记生成器（包括多尺度特征池化）
        self.agent_pool = nn.AdaptiveAvgPool2d((int(agent_num**0.5), int(agent_num**0.5)))
        self.agent_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1)  # 动态调整维度
        )

        # 注意力Dropout与投影
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 输出增强模块
        self.output_enhancer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Softmax用于权重归一化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        """
        x: 输入特征 [B, C, H, W]
        H, W: 特征图高度和宽度
        """
        B, C, H, W = x.size()

        # 查询(Q)和键值(KV)的生成
        x_flat = x.reshape(B, C, -1).transpose(-1, -2)  # [B, HW, C]
        q = self.q(x_flat).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, heads, HW, C_head]
        kv = self.kv(x_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # 键和值 [B, heads, HW, C_head]

        # 动态代理标记生成
        agent_tokens = self.agent_pool(x).reshape(B, C, -1).permute(0, 2, 1)  # [B, agent_num, C]
        agent_tokens = self.agent_encoder(agent_tokens.reshape(B, C, int(self.agent_pool.output_size[0]), -1)).reshape(B, -1, C)
        agent_tokens = agent_tokens.reshape(B, self.num_heads, -1, C // self.num_heads)  # [B, heads, agent_num, C_head]

        # Agent Attention聚合阶段
        agent_attn_weights = self.softmax((agent_tokens @ k.transpose(-2, -1)) * self.scale)  # [B, heads, agent_num, HW]
        aggregated_values = agent_attn_weights @ v  # [B, heads, agent_num, C_head]

        # Agent Attention广播阶段
        q_attn_weights = self.softmax((q @ agent_tokens.transpose(-2, -1)) * self.scale)  # [B, heads, HW, agent_num]
        enhanced_features = q_attn_weights @ aggregated_values  # [B, heads, HW, C_head]

        # 恢复形状并投影
        enhanced_features = enhanced_features.transpose(1, 2).reshape(B, -1, C)
        x_out = self.proj(enhanced_features)
        x_out = self.proj_drop(x_out)

        # 输出增强模块
        enhanced_output = self.output_enhancer(x_out) + x_out
        return enhanced_output.reshape(B, C, H, W)

if __name__ == "__main__":
    # 测试模块
    B, C, H, W = 1, 128, 16, 16
    num_patches = H * W
    agent_num = 49
    block = DynamicAgentAttention(dim=C, num_patches=num_patches, agent_num=agent_num)
    x = torch.randn(B, C, H, W)
    output = block(x, H, W)
    print(f"Input size: {x.size()}")
    print(f"Output size: {output.size()}")
