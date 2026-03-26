import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResampleDown(nn.Module):
    """步长为 2 的卷积下采样模块"""
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        # 3x3 卷积 + stride=2 做下采样，保留局部空间结构信息
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 标准的 BN + 激活，在压缩通道与空间的同时缓解梯度消失
        return self.act(self.bn(self.conv(x)))


def hw_to_seq(x):
    """(B, C, H, W) -> (B, H*W, C)，将特征图展平成序列方便做注意力"""
    B, C, H, W = x.shape
    return x.view(B, C, H * W).transpose(1, 2).contiguous()  # (B, N, C)


def seq_to_hw(x, H, W):
    """(B, N, C) -> (B, C, H, W)，将序列还原成二维特征图"""
    B, N, C = x.shape
    assert N == H * W, "N 与 H*W 不匹配，无法还原为特征图"
    return x.transpose(1, 2).contiguous().view(B, C, H, W)


class CrossScaleGlobalRelationalMixer(nn.Module):
    """
    跨尺度全局关系混合模块（Cross-Scale Global Relational Mixer, CSGRM）

    设计动机（偏 CVPR 风格）：
      1）长距离依赖：集中建模微图之间的全局关系，而不是只看局部卷积感受野。
      2）跨尺度交互：用层级式的下采样特征作为 Query，与高分辨率特征交互，模拟“粗到细”建模过程。
      3）全局语义调制：利用全局 token 生成通道注意力，对输出进行语义重标定。

    整体流程（输入输出空间尺寸不变）：
      f      : (B, C, H, W)
      fd2    : ResampleDown(f)              -> (B, C, H/2, W/2)
      fd4    : ResampleDown(fd2)           -> (B, C, H/4, W/4)

      fr_seq : hw_to_seq(f)                -> (B, N,   C)
      f2_seq : hw_to_seq(fd2)              -> (B, N/4, C)
      f4_seq : hw_to_seq(fd4)              -> (B, N/16,C)

      1）全局 token 生成：对 fr_seq 做平均得到 g，利用多头注意力从全局序列中抽取语义，再用 MLP 生成通道注意力门控。
      2）跨尺度注意力 A1：f2_seq 作为 Query，与 fr_seq 交互，得到跨尺度聚合特征 f2_seq'。
      3）跨尺度注意力 A2：f4_seq 作为 Query，与 f2_seq' 交互，进一步压缩和聚合全局信息，得到 f4_seq'。

      然后：
      f4_hw  -> 上采样到 H/2,W/2，与 fd2 相加形成中尺度融合特征
      融合特征通过局部卷积细化，再上采样回 H,W
      最后用通道注意力门控输出，并与输入 f 做残差相加。

    该模块既利用了多尺度结构，又通过全局 token 和跨尺度注意力实现了较强的全局建模能力。
    """
    def __init__(self, channels, num_heads=8, down_channels=None, up_mode="bilinear", reduction=4):
        super().__init__()
        self.C = channels
        self.up_mode = up_mode

        # 两级下采样：分别得到 1/2 和 1/4 分辨率的特征
        self.down2 = ResampleDown(channels, out_channels=down_channels or channels)
        self.down4 = ResampleDown(channels, out_channels=down_channels or channels)

        # 跨尺度多头注意力：
        #   A1：中尺度 fd2 <-> 全分辨率 f
        #   A2：小尺度 fd4 <-> 中尺度（已融合）特征
        self.mha_cross_1 = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.mha_cross_2 = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

        # 全局 token 注意力，用于构造通道级语义门控
        self.mha_global = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

        # LayerNorm 保证注意力前后分布更稳定
        self.ln_fr  = nn.LayerNorm(channels)
        self.ln_f2q = nn.LayerNorm(channels)
        self.ln_f4q = nn.LayerNorm(channels)

        # 通道注意力 MLP：基于全局 token 输出做通道级重标定
        hidden_dim = max(channels // reduction, 8)  # 防止通道数很小时维度为 0
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )

        # 跨尺度融合后的局部细化卷积：深度可分离卷积 + 1x1 PW 卷积
        self.local_refine = nn.Sequential(
            # 深度卷积：只在空间维度聚合，每个通道单独处理，保留语义解耦
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 1x1 卷积：在通道维度做线性组合，增强通道间交互
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 输出投影，将融合后的特征再做一次线性变换，方便与输入对齐
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, f):
        """
        f: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = f.shape
        assert C == self.C, "输入通道数与初始化不一致"
        assert H % 4 == 0 and W % 4 == 0, "为了简化实现，这里假设 H,W 能被 4 整除"

        # ------------------------------------------------------------------
        # 1. 多尺度下采样
        # ------------------------------------------------------------------
        fd2 = self.down2(f)   # (B, C, H/2, W/2)
        fd4 = self.down4(fd2) # (B, C, H/4, W/4)

        H2, W2 = H // 2, W // 2
        H4, W4 = H // 4, W // 4

        # ------------------------------------------------------------------
        # 2. 转换为序列形式，便于做多头注意力
        # ------------------------------------------------------------------
        fr_seq  = hw_to_seq(f)    # (B, N,   C)
        f2_seq  = hw_to_seq(fd2)  # (B, N/4, C)
        f4_seq  = hw_to_seq(fd4)  # (B, N/16,C)

        # 归一化，提升训练稳定性
        fr_norm  = self.ln_fr(fr_seq)
        f2q_norm = self.ln_f2q(f2_seq)
        f4q_norm = self.ln_f4q(f4_seq)

        # ------------------------------------------------------------------
        # 3. 全局 token 生成 + 通道注意力门控
        # ------------------------------------------------------------------
        # 对全分辨率序列做平均，相当于一个全局 token
        global_token = fr_seq.mean(dim=1, keepdim=True)  # (B, 1, C)

        # 使用多头注意力从全局序列中抽取更丰富的全局语义
        # Query: global_token, Key/Value: fr_norm
        global_token_refined, _ = self.mha_global(
            query=global_token,
            key=fr_norm,
            value=fr_norm,
            need_weights=False
        )  # (B, 1, C)

        # 将全局 token 映射成通道注意力权重，用于调制最终输出
        global_vec  = global_token_refined.squeeze(1)    # (B, C)
        channel_gate = self.channel_mlp(global_vec).view(B, C, 1, 1)  # (B, C, 1, 1)

        # ------------------------------------------------------------------
        # 4. 跨尺度注意力交互（中尺度 <-> 全分辨率）
        # ------------------------------------------------------------------
        # 中尺度序列作为 Query，向全分辨率序列拉取信息
        attn_f2, _ = self.mha_cross_1(
            query=f2q_norm,   # (B, N/4, C)
            key=fr_norm,      # (B, N,   C)
            value=fr_norm,    # (B, N,   C)
            need_weights=False
        )
        f2_seq = f2_seq + attn_f2  # 残差连接，保留原始信息

        # ------------------------------------------------------------------
        # 5. 跨尺度注意力交互（小尺度 <-> 中尺度）
        # ------------------------------------------------------------------
        # 小尺度序列作为 Query，进一步从已经融合过的中尺度序列中抽取信息
        f2_seq_norm = self.ln_f2q(f2_seq)  # 重新归一化一次
        attn_f4, _ = self.mha_cross_2(
            query=f4q_norm,     # (B, N/16, C)
            key=f2_seq_norm,    # (B, N/4,  C)
            value=f2_seq_norm,  # (B, N/4,  C)
            need_weights=False
        )
        f4_seq = f4_seq + attn_f4  # 残差

        # ------------------------------------------------------------------
        # 6. 序列还原回特征图，并做跨尺度融合 + 局部细化
        # ------------------------------------------------------------------
        # 小尺度特征图
        fd4_hw = seq_to_hw(f4_seq, H4, W4)  # (B, C, H/4, W/4)

        # 上采样到中尺度
        fd4_up = F.interpolate(
            fd4_hw,
            size=(H2, W2),
            mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None
        )

        # 与中尺度原始特征残差融合，带入跨尺度的全局信息
        f_mid = fd2 + fd4_up  # (B, C, H/2, W/2)

        # 当作一个新的中尺度特征，通过局部卷积进一步细化边缘和纹理
        f_mid_refined = self.local_refine(f_mid)  # (B, C, H/2, W/2)

        # 再上采样回原分辨率
        f_up = F.interpolate(
            f_mid_refined,
            size=(H, W),
            mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None
        )  # (B, C, H, W)

        # ------------------------------------------------------------------
        # 7. 通道注意力门控 + 残差输出
        # ------------------------------------------------------------------
        # 通道 gate 只做缩放，不改变空间分布
        f_out = self.proj_out(f_up) * channel_gate  # (B, C, H, W)

        # 最终残差连接，保持网络优化的稳定性
        out = f + f_out

        return out


# ----------------- 简单张量测试 -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造一组测试输入
    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    # 实例化跨尺度全局关系混合模块（CSGRM）
    net = CrossScaleGlobalRelationalMixer(
        channels=C,
        num_heads=8,
        down_channels=None,
        up_mode="bilinear",
        reduction=4
    ).to(device)

    y = net(x)

    print(net)
    print("\n输入张量形状 :", x.shape)
    print("输出张量形状 :", y.shape)
