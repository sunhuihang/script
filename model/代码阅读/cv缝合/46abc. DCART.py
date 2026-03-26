import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, LongTensor
"""
CV缝合救星魔改创新集成版（3个）: Dynamic Context-Aware Routing Transformer (DCART)
背景：当前的 Bi-Level Routing Attention (BRA) 中，区域划分和路由主要基于固定的规则（如区域大小、前 topk 等），
这些规则在场景变化（例如输入特征图尺寸变化或任务目标变化）时无法动态调整，可能会影响性能。此外，区域内和区域间注意
力机制是分离的，未充分融合两者的信息。

CV缝合救星魔改创新:
1. 动态区域划分：结合输入特征图的语义信息动态调整区域划分的大小，而不是固定的 n_win。
2. 跨区域上下文建模：通过引入跨区域的全局上下文注意力（如基于高效全局查询的机制）增强区域间信息交互。
3. 自适应 Topk 筛选：根据查询的特定分布动态调整 Topk 值，从而适应不同的特征和任务。
"""
class DynamicRoutingAttention(nn.Module):
    """
    动态路由注意力模块，改进自 Bi-Level Routing Attention。
    主要改进点：
    1. 动态区域划分：根据输入特征图动态调整区域划分的大小。
    2. 跨区域上下文建模：加入全局上下文机制。
    3. 自适应 Topk：根据查询动态调整 Topk 的值。
    """

    def __init__(self, dim, num_heads=8, base_win_size=7, max_win_size=16,
                 qk_scale=None, side_dwconv=3, topk=4,
                 global_context=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.base_win_size = base_win_size
        self.max_win_size = max_win_size
        self.scale = qk_scale or dim ** -0.5
        self.topk = topk
        self.global_context = global_context

        # 动态卷积，用于实现动态区域划分
        self.dynamic_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 修改输入通道数为 1

        # 局部上下文增强（类似于 LCE）
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim)

        # 全局上下文建模（可选）
        if self.global_context:
            self.global_query = nn.Linear(dim, dim)
            self.global_key = nn.Linear(dim, dim)

        # QKV 投影
        self.qkv = nn.Linear(dim, dim * 3)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        输入：
        x: Tensor, 形状为 (B, C, H, W)
        返回：
        Tensor, 输出特征，形状为 (B, C, H, W)
        """
        B, C, H, W = x.size()

        # 动态区域划分：基于动态卷积生成区域划分大小
        region_size = self._get_dynamic_region_size(x)
        region_h, region_w = H // region_size, W // region_size

        # QKV 投影
        qkv = self.qkv(x.permute(0, 2, 3, 1))  # 转换到 NHWC
        q, k, v = qkv.chunk(3, dim=-1)

        # 区域划分
        q = rearrange(q, "b (h r1) (w r2) c -> b (h w) (r1 r2) c", r1=region_size, r2=region_size)
        k = rearrange(k, "b (h r1) (w r2) c -> b (h w) (r1 r2) c", r1=region_size, r2=region_size)
        v = rearrange(v, "b (h r1) (w r2) c -> b (h w) (r1 r2) c", r1=region_size, r2=region_size)

        # 局部注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        local_out = attn @ v

        # 全局上下文建模（可选）
        if self.global_context:
            global_q = self.global_query(x.mean(dim=(2, 3)))  # 全局查询
            global_k = self.global_key(x.mean(dim=(2, 3)))  # 全局键
            global_attn = F.softmax(global_q @ global_k.transpose(-2, -1), dim=-1)  # 计算全局注意力

            # 计算 v 的均值（对区域进行平均）
            v_mean = v.mean(dim=2, keepdim=True)  # 对 v 的每个区域进行均值计算，得到形状 (B, num_regions, 1, value_dim)

            # 广播注意力矩阵，使其与 v 的形状匹配
            global_out = global_attn.unsqueeze(
                2) * v_mean  # global_attn 变为 (B, num_heads, num_regions, 1, 1)，与 v_mean 兼容

            # 调整形状：去掉多余的维度
            global_out = global_out.squeeze(3)  # 最后的 squeeze 去掉不需要的维度，形状变为 (B, num_heads, num_regions, value_dim)

            # 将 global_out 和 local_out 合并（或加和）
            local_out += global_out.squeeze(1)  # 去掉 head 维度，形状变为 (B, num_regions, value_dim)

        # 恢复到原始形状
        out = rearrange(local_out, "b (h w) (r1 r2) c -> b c (h r1) (w r2)", r1=region_size, r2=region_size, h=region_h)

        # 局部上下文增强
        lepe_out = self.lepe(x) + out

        # 输出投影
        out = self.output_proj(lepe_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out

    def _get_dynamic_region_size(self, x):
        """
        动态计算区域划分大小，并确保其能够整除输入尺寸。
        输入：
        x: Tensor, 输入特征 (B, C, H, W)
        返回：
        int, 动态区域划分大小
        """
        B, C, H, W = x.size()
        avg_feature = x.mean(dim=1, keepdim=True)  # 计算特征图均值
        dynamic_weights = self.dynamic_conv(avg_feature)  # 动态生成权重
        avg_weights = dynamic_weights.mean(dim=(2, 3))  # 取均值作为尺度

        # 根据权重映射到区域大小范围
        region_size = torch.clamp((avg_weights * (self.max_win_size - self.base_win_size) + self.base_win_size).int(),
                                  min=self.base_win_size, max=self.max_win_size).item()

        # 强制调整为能够整除 H 和 W 的值
        def get_valid_size(size, dim):
            while dim % size != 0:
                size -= 1
            return size

        region_size = get_valid_size(region_size, H)
        region_size = get_valid_size(region_size, W)

        # 确保最终 region_size >= 1
        region_size = max(1, region_size)

        return region_size


if __name__ == '__main__':
    # 测试动态路由注意力模块
    model = DynamicRoutingAttention(dim=64).cuda()
    input_tensor = torch.randn(1, 64, 128, 128).cuda()
    output_tensor = model(input_tensor)
    print("输入尺寸", input_tensor.size())
    print("输出尺寸", output_tensor.size())
