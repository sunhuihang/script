# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# -----------------------------
# 工具模块：DropPath（Stochastic Depth）
# -----------------------------
class DropPath(nn.Module):
    """逐样本的随机深度（stochastic depth）。训练时按比例丢弃残差分支。"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 以 batch 为粒度生成掩码，并在空间上广播
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0/1
        return x.div(keep_prob) * random_tensor


# -----------------------------
# 基础模块
# -----------------------------
class Residual(nn.Module):
    """简单残差封装：y = x + f(x)"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvUtr(nn.Module):
    """
    轻量化卷积化 Transformer 嵌入：
    - 大核深度可分离卷积模拟大感受野
    - 1x1 倒残差通道交互
    """
    def __init__(self, ch_in, ch_out, depth=1, kernel=3):
        super(ConvUtr, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(kernel, kernel),
                              groups=ch_in, padding=(kernel // 2, kernel // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
            ) for _ in range(depth)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class Embeddings(nn.Module):
    """Stem + 三阶段 ConvUtr 编码，符合论文的层数与大核设计"""
    def __init__(self, inch=3, dims=[8, 16, 32], depths=[1, 1, 3], kernels=[3, 3, 7]):
        super(Embeddings, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inch, dims[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ConvUtr(dims[0], dims[0], depth=depths[0], kernel=kernels[0])
        self.layer2 = ConvUtr(dims[0], dims[1], depth=depths[1], kernel=kernels[1])
        self.layer3 = ConvUtr(dims[1], dims[2], depth=depths[2], kernel=kernels[2])
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x0 = self.stem(x)         # H, W
        x0 = self.layer1(x0)      # H, W

        x1 = self.down(x0)        # H/2, W/2
        x1 = self.layer2(x1)

        x2 = self.down(x1)        # H/4, W/4
        x2 = self.layer3(x2)

        return x2, (x0, x1, x2)


# -----------------------------
# MLP（token & conv 两种形态）
# -----------------------------
class Mlp(nn.Module):
    """标准 token MLP，用于注意力后的通道混合"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class CMlp(nn.Module):
    """卷积版 MLP：用深度可分离卷积替代全连接，更利于空间结构保持"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, padding=1, groups=in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, padding=1, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# -----------------------------
# 全局路径：稀疏注意力（带下采样聚合）
# -----------------------------
class GlobalSparseAttn(nn.Module):
    """
    稀疏全局注意力：
    - 可选的空间下采样（sr_ratio>1 时）进行 token 聚合，降低复杂度
    - 标准多头自注意力
    - 可选的转置卷积恢复回原分辨率
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            # 平均池化下采样，卷积上采样
            self.sampler = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
            self.local_up = nn.ConvTranspose2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.local_up = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        """
        输入：x [B, N, C]，其中 N = H*W
        输出：y [B, N, C]
        """
        B, N, C = x.shape
        if self.sr_ratio > 1:
            # 将 token 还原为 feature map，再做平均池化聚合
            fmap = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
            fmap = self.sampler(fmap)                     # [B, C, H/sr, W/sr]
            x = fmap.flatten(2).transpose(1, 2)           # [B, N', C]

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, Nt, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr_ratio > 1:
            # 将 token 还原为 fmap，再用深度转置卷积回到原分辨率
            h, w = H // self.sr_ratio, W // self.sr_ratio
            fmap = x.permute(0, 2, 1).reshape(B, C, h, w)
            fmap = self.local_up(fmap)                    # [B, C, H, W]
            x = fmap.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------
# 局部路径：SK-LKA（多核大卷积选择性聚合）
# -----------------------------
class SKLargeKernelAgg(nn.Module):
    """
    Selective-Kernel Large-Kernel Aggregation：
    - 三个深度可分离大核卷积分支（5/9/17）
    - SE 风格通道注意力产生分支权重（softmax），进行选择融合
    """
    def __init__(self, dim):
        super().__init__()
        self.dw5  = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw9  = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.dw17 = nn.Conv2d(dim, dim, 17, padding=8, groups=dim)

        self.bn = nn.BatchNorm2d(dim)

        # 通道注意力产生三个分支的权重
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, dim // 4 if dim >= 4 else 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(dim // 4 if dim >= 4 else 1, dim * 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # 在“分支维度”做 softmax，见 forward

        # 融合后的通道混合
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # 三个分支
        x5  = self.dw5(x)
        x9  = self.dw9(x)
        x17 = self.dw17(x)

        # 计算分支权重
        u = self.gap(x)
        u = self.fc1(u)
        u = self.relu(u)
        u = self.fc2(u)              # [B, 3C, 1, 1]
        B, _, _, _ = u.shape
        # 重排为 [B, 3, C, 1, 1]，对“3”做 softmax 得到各分支权重
        attn = u.view(B, 3, -1, 1, 1)
        attn = self.softmax(attn)

        w5, w9, w17 = attn[:, 0], attn[:, 1], attn[:, 2]  # [B, C, 1, 1]
        out = w5 * x5 + w9 * x9 + w17 * x17
        out = self.bn(out)
        out = self.fuse(out)
        return out


# -----------------------------
# 新增：GLSR Block（全局-局部选择式路由）
# -----------------------------
class GLSRBlock(nn.Module):
    """
    输入输出均为 [B, C, H, W]
    结构：
      Local Path : SK-LKA -> BN -> CMlp（卷积版）
      Global Path: pos_embed -> LN -> GlobalSparseAttn(sparse) -> reshape
      Selective Routing: 基于 [Local, Global] 逐像素 gate 融合
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=2):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # 局部分支
        self.local_agg = SKLargeKernelAgg(dim)
        self.local_bn  = nn.BatchNorm2d(dim)
        self.local_mlp = CMlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # 全局分支
        self.norm1 = norm_layer(dim)
        self.attn  = GlobalSparseAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        self.token_mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # 选择式路由 gate
        self.gate_gen = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.pos_embed(x)  # 轻量位置编码

        # ---------- 局部分支 ----------
        l = self.local_agg(x)
        l = self.local_bn(l)
        l = l + self.drop_path(self.local_mlp(l))

        # ---------- 全局分支 ----------
        t = x.flatten(2).transpose(1, 2)      # [B, N, C]
        t = t + self.drop_path(self.attn(self.norm1(t), H, W))
        t = t + self.drop_path(self.token_mlp(self.norm2(t)))
        g = t.transpose(1, 2).reshape(B, C, H, W)

        # ---------- 选择式路由融合 ----------
        gate = torch.sigmoid(self.gate_gen(torch.cat([l, g], dim=1)))  # [B, C, H, W]
        out = gate * l + (1.0 - gate) * g
        return out


# -----------------------------
# 解码端：上采样与门控对齐
# -----------------------------
class up_conv(nn.Module):
    """双线性上采样 + 3x3 Conv"""
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class conv_block(nn.Module):
    """两次 3x3 Conv 的标准卷积块"""
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class GateAlign(nn.Module):
    """
    门控对齐：
    - 在与解码特征拼接之前，对 skip 进行逐像素抑噪加权
    - 输入：skip（已对齐分辨率）, dec 当前解码特征
    - 输出：gated_skip
    """
    def __init__(self, ch_skip, ch_dec):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch_skip + ch_dec, ch_skip, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_skip, ch_skip, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, skip, dec):
        gate = self.gate(torch.cat([skip, dec], dim=1))
        return skip * gate


# -----------------------------
# 主干：GLSR-MobileUViT
# -----------------------------
class GLSRMobileUViT(nn.Module):
    """
    总体结构：
      Encoder: Stem + ConvUtr×3
      Bottleneck: GLSRBlock（sr_ratio=2）×L4 -> 通道扩展 -> GLSRBlock（sr_ratio=1）×L5 -> 通道回降
      Decoder: 门控对齐 + U 形逐级上采样
    """
    def __init__(self,
                 inch=3,
                 dims=[16, 32, 64, 128],
                 depths=[1, 1, 3, 3, 3],       # 前三段 ConvUtr；接着 GLSR 的两个深度
                 kernels=[3, 3, 7],
                 embed_dim=256,               # 用于计算注意力头数
                 out_channel=1):
        super(GLSRMobileUViT, self).__init__()

        # 编码（与原始 Mobile U-ViT 对齐）
        self.patch_embeddings = Embeddings(inch=inch, dims=dims, depths=depths[:3], kernels=kernels)

        # GLSR 瓶颈（sr_ratio=2：先做 token 聚合）
        num_heads = max(1, embed_dim // 64)
        self.glsr_bottleneck = nn.ModuleList([
            GLSRBlock(dim=dims[2], num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                      drop=0., attn_drop=0., drop_path=0.1, sr_ratio=2)
            for _ in range(depths[3])
        ])

        # 通道扩展（与论文中 bottleneck 过渡保持一致风格）
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.expend_dims = conv_block(ch_in=dims[2], ch_out=dims[3])

        # 顶部 Transformer 风格的 GLSR（sr_ratio=1：不再下采样，纯全局）
        self.top_bottleneck = nn.ModuleList([
            GLSRBlock(dim=dims[3], num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                      drop=0., attn_drop=0., drop_path=0.0, sr_ratio=1)
            for _ in range(depths[4])
        ])

        # 通道回降
        self.reduce_dims = conv_block(ch_in=dims[3], ch_out=dims[2])

        # 解码 + 门控对齐
        self.align2 = GateAlign(ch_skip=dims[2], ch_dec=dims[2])
        self.align1 = GateAlign(ch_skip=dims[1], ch_dec=dims[1])
        self.align0 = GateAlign(ch_skip=dims[0], ch_dec=dims[0])

        self.Up_conv4 = conv_block(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = conv_block(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = conv_block(ch_in=dims[0] * 2, ch_out=dims[0])

        # 轻量尾部
        self.Up1 = up_conv(ch_in=dims[0], ch_out=dims[0])
        self.head = nn.Conv2d(dims[0], out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 单通道图像重复到 3 通道以复用 stem
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # 编码
        x, skip = self.patch_embeddings(x)  # x: [B, C2, H/4, W/4]; skip: (x0, x1, x2)
        x = self.down(x)                    # 进一步下采样，进入瓶颈 H/8, W/8

        # GLSR 瓶颈（sr_ratio=2）
        for blk in self.glsr_bottleneck:
            x = blk(x)

        # 通道扩展
        x = self.expend_dims(x)

        # 顶部 GLSR（sr_ratio=1）
        for blk in self.top_bottleneck:
            x = blk(x)

        # 通道回降，准备解码
        x = self.reduce_dims(x)

        # 解码：逐级与 skip 融合（加入门控对齐以抑制噪声 skip）
        s2 = self.down(skip[2])                   # 与 x 对齐的下采样
        s2 = self.align2(s2, x)
        x = self.Up_conv4(torch.cat((x, s2), dim=1))

        x = self.Up3(x)
        s1 = self.down(skip[1])
        s1 = self.align1(s1, x)
        x = self.Up_conv3(torch.cat((x, s1), dim=1))

        x = self.Up2(x)
        s0 = self.down(skip[0])
        s0 = self.align0(s0, x)
        x = self.Up_conv2(torch.cat((x, s0), dim=1))

        x = self.Up1(x)
        x = self.head(x)
        return x


# -----------------------------
# 便捷构造函数
# -----------------------------
def glsr_mobileuvit(inch=3, dims=[16, 32, 64, 128], depths=[1, 1, 3, 3, 3],
                    kernels=[3, 3, 7], embed_dim=256, out_channel=1):
    """
    推荐配置：与 Mobile U-ViT Base 一致的通道与深度，方便横向对比
    """
    return GLSRMobileUViT(inch=inch, dims=dims, depths=depths,
                          kernels=kernels, embed_dim=embed_dim, out_channel=out_channel)


def glsr_mobileuvit_l(inch=3, dims=[32, 64, 128, 256], depths=[1, 1, 3, 3, 4],
                      kernels=[3, 3, 7], embed_dim=512, out_channel=1):
    """
    Large 版本：更宽通道与更深顶部瓶颈
    """
    return GLSRMobileUViT(inch=inch, dims=dims, depths=depths,
                          kernels=kernels, embed_dim=embed_dim, out_channel=out_channel)


# -----------------------------
# 简单自检
# -----------------------------
if __name__ == "__main__":
    # 构造一张随机图：B=1, C=3, H=W=64（方便 demo）
    x = torch.randn(1, 3, 64, 64)

    # 初始化模型（Base）
    model = glsr_mobileuvit(inch=3,
                            dims=[16, 32, 64, 128],
                            depths=[1, 1, 3, 3, 3],
                            kernels=[3, 3, 7],
                            embed_dim=256,
                            out_channel=1)

    # 前向传播
    with torch.no_grad():
        y = model(x)

    # 打印结构与形状
    print(model)
    print("\n[GLSR-MobileUViT] 运行自检")
    print("输入张量形状:", x.shape)       # [B, C, H, W]
    print("\n微信公众号:CV缝合救星\n")
    print("输出张量形状:", y.shape)       # [B, C, H, W]
