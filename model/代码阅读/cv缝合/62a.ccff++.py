import torch
import torch.nn as nn
# CCFF++for Perry

# 激活函数选择器
def get_activation(act: str, inplace=True):
    act = act.lower()
    if act == 'silu': return nn.SiLU(inplace=inplace)
    if act == 'relu': return nn.ReLU(inplace=inplace)
    if act == 'leaky_relu': return nn.LeakyReLU(inplace=inplace)
    if act == 'gelu': return nn.GELU()
    if act is None: return nn.Identity()
    raise RuntimeError(f"Unsupported activation: {act}")

# SE注意力
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(self.avg_pool(x))
        return x * weight

# Depthwise+Pointwise+SE轻量模块
class LightweightBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='silu'):
        super().__init__()
        self.dw = nn.Conv2d(ch_in, ch_in, 3, padding=1, groups=ch_in, bias=False)
        self.pw = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)
        self.se = SEModule(ch_out)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x

# 动态门控融合（cross gating）
class CrossGatingFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self, x1, x2):
        w1 = self.gate1(x2)
        w2 = self.gate2(x1)
        return x1 * w1 + x2 * w2

# 魔改CCFF++模块
class CCFFPlus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 fusion_mode="add",
                 act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(*[
            LightweightBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        self.fusion = CrossGatingFusion(hidden_channels)
        self.mode = fusion_mode
        if fusion_mode == "concat":
            self.post_fuse = nn.Conv2d(hidden_channels * 2, out_channels, 1)
        else:
            self.post_fuse = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1 = self.blocks(x1)
        x2 = self.blocks(x2)
        fused = self.fusion(x1, x2)
        if self.mode == "concat":
            fused = self.post_fuse(torch.cat([x1, x2], dim=1))
        else:
            fused = self.post_fuse(fused)
        return fused

if __name__ == '__main__':
    # 配置参数
    in_channels = 64
    out_channels = 32
    H, W = 32, 32  # 特征图尺寸
    batch_size = 1

    # 构建模型
    model = CCFFPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=2,
        expansion=1.0,
        fusion_mode="add",  # 可选 "add" 或 "concat"
        act="silu"
    )

    # 打印模型结构
    print(model)

    # 随机生成两个输入特征图（模拟来自不同尺度的输出）
    input1 = torch.randn(batch_size, in_channels, H, W)
    input2 = torch.randn(batch_size, in_channels, H, W)

    # 前向推理
    output = model(input1, input2)

    # 输出尺寸信息
    print("Input1 shape:", input1.shape)
    print("Input2 shape:", input2.shape)
    print("Output shape:", output.shape)
