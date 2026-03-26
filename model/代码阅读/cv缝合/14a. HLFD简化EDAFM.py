
"""
Efficient Dual-Attention Fusion Module (EDAFM)
高效性（Efficient）、双分支结构（Dual）、注意力机制（Attention）和特征融合（Fusion）

缺点：高低频分解和重构过程中依赖小波变换和逆卷积操作。这种设计尽管有助于分离不同频率的信息，但在实际应用中存在信息整合不充分的问题
，导致低频结构和高频细节可能无法在最终输出中实现最优融合，影响了图像恢复效果的整体平衡。

CV缝合救星魔改：引入 SimpleAttention 和 DualBranchFusion 模块，这个改进版模型实现了更高效的特征提取和频率信息处理。
SimpleAttention通过通道注意力机制增强了模型的特征选择能力，DualBranchFusion则利用普通卷积和膨胀卷积的双分支结构来捕获多尺度特征，
从而替代了原有的复杂小波变换。最终的 EnhancedHLFD 模块将这两者结合，使模型在提升图像恢复效果的同时，保持结构简洁和稳定，更易于维护。
"""
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, channels):
        super(SimpleAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.avg_pool(x)
        weights = self.fc(weights)
        return x * weights

class DualBranchFusion(nn.Module):
    def __init__(self, channels):
        super(DualBranchFusion, self).__init__()
        self.conv_standard = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_dilated = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False)

    def forward(self, x):
        standard_out = self.conv_standard(x)
        dilated_out = self.conv_dilated(x)
        return standard_out + dilated_out

class MFDAFM(nn.Module):
    def __init__(self, channels):
        super(MFDAFM, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2)
        self.attention = SimpleAttention(channels)
        self.dual_branch_fusion = DualBranchFusion(channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # 多频分解
        low_freq = self.down(x)
        high_freq = x - F.interpolate(low_freq, size=x.size()[-2:], mode='bilinear', align_corners=True)

        # 低频特征通过注意力增强
        low_freq = F.interpolate(self.attention(low_freq), size=x.size()[-2:], mode='bilinear', align_corners=True)

        # 高频特征通过双分支卷积增强
        high_freq = self.dual_branch_fusion(high_freq)

        # 低频和高频特征融合
        fused = low_freq + high_freq
        out = self.conv_out(fused)
        return out

# 示例代码：实例化模块并进行前向传播
if __name__ == "__main__":
    model = MFDAFM(channels=32)
    input_tensor = torch.randn(1, 32, 64, 64)
    output = model(input_tensor)
    print("输入尺寸:", input_tensor.size())
    print("输出尺寸:", output.size())


