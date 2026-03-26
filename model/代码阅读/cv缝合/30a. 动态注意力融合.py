import torch
import torch.nn as nn
"""
CV缝合救星魔改创新：动态注意力融合（Dynamic Attention Fusion）
1. 创新背景：
传统的多模态融合方式大多采用固定的融合流程（例如直接拼接或串联注意力模块），在处理多模态任务时可能
会因为模态间信息分布的不均衡而导致性能下降。
2. 创新实现：Dynamic Attention Fusion
动态注意力融合机制通过一个共享查询模块提取全局特征，分别对 RGB 和深度模态生成键值对特征，随后基于
动态生成的注意力权重矩阵，对模态间的特征进行自适应融合。
具体实现细节如下：
A. 共享查询模块 (Shared Query Module)： 从 RGB 和深度特征中提取共享的全局上下文信息，用于指导
模态间的注意力计算。
B. 动态注意力生成 (Dynamic Attention Generation)： 分别对 RGB 和深度模态计算键值特征，并通过
点积操作生成注意力权重矩阵，动态调整各模态的特征贡献。
C. 融合机制 (Fusion Mechanism)： 利用生成的注意力权重对 RGB 和深度特征进行加权求和，生成融合特
征。最终，通过一个控制参数 gamma 调节融合特征的输出强度。
"""
# ------------------ 基础卷积模块 ------------------
class BBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BBasicConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes,
                kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basicconv(x)

# ------------------ 上下文感知 DenseLayer ------------------
class ContextAwareDenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=2, num_heads=4):
        super(ContextAwareDenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        # 下采样通道
        self.down = nn.Conv2d(in_C, mid_C, 1)

        # DenseBlock 部分
        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, padding=1))

        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=mid_C, num_heads=num_heads, batch_first=True)

        # 最终融合
        self.fuse = BBasicConv2d(in_C + mid_C, out_C, 3, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)

        # DenseBlock 计算
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        # 转换形状用于注意力计算
        B, C, H, W = down_feats.size()
        flat_feats = down_feats.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # 自注意力
        context_feats, _ = self.attention(flat_feats, flat_feats, flat_feats)  # (B, H*W, C)
        context_feats = context_feats.permute(0, 2, 1).view(B, C, H, W)

        # 最终特征融合
        fused_feats = torch.cat((in_feat, context_feats), dim=1)
        return self.fuse(fused_feats)

# ------------------ 动态融合 GEFM ------------------
class DynamicGEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(DynamicGEFM, self).__init__()
        self.shared_q = BBasicConv2d(in_C, out_C, 3, padding=1)
        self.rgb_kv = nn.Sequential(BBasicConv2d(out_C, out_C, 3, padding=1))
        self.depth_kv = nn.Sequential(BBasicConv2d(out_C, out_C, 3, padding=1))

        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb, depth):
        # 提取共享查询特征
        Q = self.shared_q(torch.cat([rgb, depth], dim=1))  # Q: (B, out_C, H, W)

        # RGB 特征注意力
        rgb_k = self.rgb_kv(rgb)  # (B, out_C, H, W)
        rgb_v = self.rgb_kv(rgb)  # (B, out_C, H, W)

        # 展平特征：从 (B, C, H, W) -> (B, H*W, C) 或 (B, C, H*W)
        B, C, H, W = rgb_k.size()
        rgb_k = rgb_k.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        rgb_v = rgb_v.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        Q_flat = Q.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # 计算注意力权重
        rgb_attention = self.softmax(torch.bmm(Q_flat, rgb_k.permute(0, 2, 1)))  # (B, H*W, H*W)
        rgb_refined = torch.bmm(rgb_attention, rgb_v).permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        # 深度特征注意力
        depth_k = self.depth_kv(depth)  # (B, out_C, H, W)
        depth_v = self.depth_kv(depth)  # (B, out_C, H, W)
        depth_k = depth_k.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        depth_v = depth_v.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        depth_attention = self.softmax(torch.bmm(Q_flat, depth_k.permute(0, 2, 1)))  # (B, H*W, H*W)
        depth_refined = torch.bmm(depth_attention, depth_v).permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        # 动态加权融合
        fused = self.gamma * (rgb_refined + depth_refined)
        return fused


# ------------------ 升级版 PSFM ------------------
class UpgradedPSFM(nn.Module):
    def __init__(self, Channel):
        super(UpgradedPSFM, self).__init__()
        self.rgb_dense = ContextAwareDenseLayer(Channel, Channel)
        self.depth_dense = ContextAwareDenseLayer(Channel, Channel)
        self.dynamic_fusion = DynamicGEFM(Channel * 2, Channel)

    def forward(self, data):
        rgb, depth = data
        rgb_feats = self.rgb_dense(rgb)
        depth_feats = self.depth_dense(depth)
        fused_feats = self.dynamic_fusion(rgb_feats, depth_feats)
        return fused_feats


# ------------------ 测试 ------------------
if __name__ == "__main__":
    block = UpgradedPSFM(Channel=32)
    input1 = torch.rand(1, 32, 64, 64)  # RGB 特征
    input2 = torch.rand(1, 32, 64, 64)  # 深度特征
    output = block([input1, input2])
    print("Input1 size:", input1.size())
    print("Input2 size:", input2.size())
    print("Output size:", output.size())
