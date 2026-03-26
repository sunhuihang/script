
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResampleDown(nn.Module):
    """步长2的卷积下采样"""
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))                                                                                                                                                                     # 微信公众号:CV缝合救星


def hw_to_seq(x):
    """(B,C,H,W) -> (B, H*W, C)"""
    B, C, H, W = x.shape
    return x.view(B, C, H * W).transpose(1, 2).contiguous()  # (B,N,C)                                                                                                                                                                     # 微信公众号:CV缝合救星


def seq_to_hw(x, H, W):
    """(B, N, C) -> (B,C,H,W)"""
    B, N, C = x.shape
    assert N == H * W
    return x.transpose(1, 2).contiguous().view(B, C, H, W)                                                                                                                                                                     # 微信公众号:CV缝合救星


class GlobalAttention(nn.Module):
    """
    两次全局多头注意力：
      fd  = downsample(f)
      fr  = reshape(f)    -> (B,N,C)
      fdr = reshape(fd)   -> (B,N/4,C)
      fd1 = MHA(q=fdr, k=fr, v=fr) + fdr
      fd2 = MHA(q=fd1, k=fr, v=fr) + fd1
      out = upsample(reshape(fd2)) + f
    输入输出都是 (B,C,H,W)，空间尺寸不变
    """
    def __init__(self, channels, num_heads=8, down_channels=None, up_mode="bilinear"):                                                                                                                                                                     # 微信公众号:CV缝合救星
        super().__init__()
        self.C = channels
        self.down = ResampleDown(channels, out_channels=down_channels or channels)
        self.mha1 = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)                                                                                                                                                                     # 微信公众号:CV缝合救星
        self.mha2 = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)                                                                                                                                                                     # 微信公众号:CV缝合救星
        self.ln_q1 = nn.LayerNorm(channels)
        self.ln_q2 = nn.LayerNorm(channels)
        self.ln_fr = nn.LayerNorm(channels)
        self.up_mode = up_mode
        # 输出前的 1x1 调整（可选，确保通道和输入一致；这里保持相同）
        self.proj_out = nn.Identity()

    def forward(self, f):
        B, C, H, W = f.shape
        assert C == self.C, "channel mismatch"

        # 下采样
        fd = self.down(f)                    # (B,C,H/2,W/2)

        # 重塑为序列
        fr  = hw_to_seq(f)                   # (B, N,   C)
        fdr = hw_to_seq(fd)                  # (B, N/4, C)

        # 预归一化（更稳定）
        fr_n  = self.ln_fr(fr)
        fdr_n = self.ln_q1(fdr)

        # 第一次全局注意力：q 来自低分辨率，k/v 来自原分辨率
        attn1, _ = self.mha1(query=fdr_n, key=fr_n, value=fr_n, need_weights=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        fd1 = fdr + attn1  # 残差

        # 第二次全局注意力：再聚合一次，得到更紧凑的全局表示
        fd1_n = self.ln_q2(fd1)
        attn2, _ = self.mha2(query=fd1_n, key=fr_n, value=fr_n, need_weights=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        fd2 = fd1 + attn2  # 残差

        # 还原形状 -> 上采样到原分辨率 -> 与输入残差相加
        fd2_hw = seq_to_hw(fd2, H // 2, W // 2)   # (B,C,H/2,W/2)
        fd2_up = F.interpolate(fd2_hw, size=(H, W), mode=self.up_mode, align_corners=False if self.up_mode=="bilinear" else None)                                                                                                                                                                     # 微信公众号:CV缝合救星
        fo = self.proj_out(fd2_up) + f
        return fo


# ----------------- 张量测试 -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    net = GlobalAttention(channels=C, num_heads=8).to(device)                                                                                                                                                                     # 微信公众号:CV缝合救星
    y = net(x)

    print(net)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家复现! \n")
    print("Input :", x.shape)
    print("Output:", y.shape)
