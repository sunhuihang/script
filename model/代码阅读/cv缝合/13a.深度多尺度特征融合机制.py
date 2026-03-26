import torch
import torch.nn as nn
import torch.nn.functional as F
"""
缺点1：局部特征与非局部特征的耦合性不足
SMFA模块的非局部特征和局部特征分别由EASA和LDE分支捕捉，虽然在最后进行融合，但这种简单的相加可能无法充分利用局部
和非局部特征之间的关系。

CV缝合救星改进思路:
在EASA和LDE分支的输出之间添加一个注意力机制，使得局部特征在非局部特征的引导下被加权，以增强局部和非局部特征的耦合
效果。使用通道注意力机制来动态调整每个通道的重要性。
"""
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class SMFAEnhanced(nn.Module):
    def __init__(self, dim=36):
        super(SMFAEnhanced, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w), mode='nearest')
        y_d = self.lde(y)

        # 将局部和非局部特征通过通道注意力机制结合
        fusion = x_l + y_d
        fusion = fusion * self.channel_attention(fusion)

        return self.linear_2(fusion)

# 测试代码
if __name__ == '__main__':
    input_tensor = torch.rand(1, 32, 256, 256)
    model = SMFAEnhanced(dim=32)
    output = model(input_tensor)
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
