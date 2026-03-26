import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 工具函数：自动 padding
# =========================
def autopad(k, p=None, d=1):
    # 将卷积输出尺寸尽量保持为 same（不改变 H/W）
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# =========================
# 激活函数：h-sigmoid / h-swish
# =========================
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# =========================
# CoordAttention
# =========================
class CoordAttiton(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
        # 分别沿 H/W 做一维池化，保留坐标方向信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 高度方向池化：B,C,H,1
        x_h = self.pool_h(x)
        # 宽度方向池化：B,C,1,W -> B,C,W,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接后用 1x1 做压缩与混合
        y = torch.cat([x_h, x_w], dim=2)  # B,C,H+W,1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 切分回 H 与 W 两段
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # B,C,1,W

        # 生成方向注意力
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 同时施加到输入
        out = identity * a_w * a_h
        return out

# =========================
# 基础卷积块：Conv+BN+ReLU
# =========================
class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# =========================
# 空间注意力：CBAM 风格
# =========================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维做 avg/max，再卷积得到空间权重图
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(x_cat))
        return attn * x_source

# ==========================================================
# CVPR风格改造模块：HF-IGate
# ==========================================================
class HF_IGate(nn.Module):
    """
    模块名称（CVPR风格）：
    HF-IGate：High-Frequency Interactive Gating Fusion
    中文名称：高频交互门控融合模块

    核心设计：
    1) 高频细节增强：使用逐通道的拉普拉斯高通滤波增强边缘/细粒度变化
    2) 双向交互门控：低层细节引导高层语义、高层语义反向约束低层细节
    3) 通道级融合图：由两路特征共同生成通道级权重图，增强融合表达能力
    4) 坐标注意力收敛：利用 CoordAttention 保留方向位置信息，提高稳定性

    输入：
      x_low  : (B, C_low,  H, W)
      x_high : (B, C_high, h, w)  (若尺寸不同会自动上采样到 H,W)
    输出：
      out    : (B, out_channel, H, W)
    """
    def __init__(self, feature_low_channel, feature_high_channel, out_channel, kernel_size=3):
        super().__init__()

        # 1) 通道对齐：两路都映射到 out_channel，便于后续融合
        self.conv_low_1x1 = CBR(feature_low_channel, out_channel, 1)
        self.conv_high_1x1 = CBR(feature_high_channel, out_channel, 1)

        # 2) 空间注意力：增强空间显著区域
        self.low_sa = SpatialAttention()
        self.high_sa = SpatialAttention()

        # 3) 高频增强：固定拉普拉斯核做逐通道高通
        lap = torch.tensor([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap)

        # 高频分支强度缩放参数（可学习），避免高频扰动过强
        self.hf_scale_low = nn.Parameter(torch.tensor(0.5))
        self.hf_scale_high = nn.Parameter(torch.tensor(0.5))

        # 4) 双向交互门控：用 GAP + 轻量 MLP 生成通道门控系数
        hidden = max(out_channel // 4, 16)
        self.gate_l2h = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel * 2, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channel, 1, bias=True),
            nn.Sigmoid()
        )
        self.gate_h2l = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel * 2, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channel, 1, bias=True),
            nn.Sigmoid()
        )

        # 5) 通道级融合图：由两路共同生成 (B,C,H,W) 的权重图
        self.mix_map = nn.Sequential(
            CBR(out_channel * 2, out_channel, kernel_size),
            nn.Conv2d(out_channel, out_channel, 1, padding=0),
            nn.Sigmoid()
        )

        # 6) 坐标注意力：增强方向敏感性
        self.ca = CoordAttiton(out_channel, out_channel)

        # 7) 输出融合：拼接后做两次轻量卷积整形
        self.fuse = nn.Sequential(
            CBR(out_channel * 2, out_channel, 1),
            CBR(out_channel, out_channel, kernel_size)
        )

    def _high_freq(self, x):
        """
        逐通道拉普拉斯高通滤波：
        - 使用 groups=C 的卷积实现每个通道独立滤波
        - 输出与输入同尺寸
        """
        B, C, H, W = x.shape
        weight = self.lap_kernel.repeat(C, 1, 1, 1)  # (C,1,3,3)
        return F.conv2d(x, weight, bias=None, stride=1, padding=1, groups=C)

    def forward(self, x_low, x_high):
        # 1) 分辨率对齐：将高层特征上采样到低层尺寸
        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode="bilinear", align_corners=True)

        # 2) 通道对齐
        low = self.conv_low_1x1(x_low)
        high = self.conv_high_1x1(x_high)

        # 3) 保留残差分支，避免注意力/高频破坏整体分布
        low_res = low
        high_res = high

        # 4) 空间注意力增强
        low_sa = self.low_sa(low)
        high_sa = self.high_sa(high)

        # 5) 高频增强（残差式叠加）
        low_hf = low_sa + self.hf_scale_low * self._high_freq(low_sa)
        high_hf = high_sa + self.hf_scale_high * self._high_freq(high_sa)

        # 6) 双向交互门控
        # 低层引导高层：根据 [high_hf, low_hf] 生成 gate_l2h，再调制 high
        gate_l2h = self.gate_l2h(torch.cat([high_hf, low_hf], dim=1))
        # 高层引导低层：根据 [low_hf, high_hf] 生成 gate_h2l，再调制 low
        gate_h2l = self.gate_h2l(torch.cat([low_hf, high_hf], dim=1))

        high_guided = high_hf * gate_l2h + high_res
        low_guided = low_hf * gate_h2l + low_res

        # 7) 通道级融合图：用两路融合后的特征共同产生权重图
        mix = torch.cat([low_guided, high_guided], dim=1)
        mix_att = self.mix_map(mix)  # (B,C,H,W)

        # 8) CoordAttention 收敛空间方向信息，并作为统一门控系数
        ca_feat = torch.sigmoid(self.ca(mix_att * (low_guided + high_guided)))

        # 9) 最终融合输出
        out = self.fuse(torch.cat([low_guided * ca_feat, high_guided * ca_feat], dim=1))
        return out

# =========================
# 直接可运行的测试
# =========================
if __name__ == '__main__':
    # 构造输入：B,C,H,W
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)

    # 实例化模块
    model = HF_IGate(feature_low_channel=32, feature_high_channel=32, out_channel=32)

    # 前向推理
    output = model(input1, input2)

    # 打印形状
    print('HF-IGate_input1_size:', input1.size())
    print('HF-IGate_input2_size:', input2.size())
    print('HF-IGate_output_size:', output.size())
    print('Module Name: HF-IGate (High-Frequency Interactive Gating Fusion)')
