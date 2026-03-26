import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=16, alpha=0.9):
        super(AdaptiveCoordAtt, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)  # 瓶颈层通道数
        self.alpha=alpha

        # Shared MLP: Conv1x1 -> BN -> ReLU
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),                                                                                                                      # 微信公众号:CV缝合救星
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        # Conv to recover channel for H and W branches
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)                                                                                                                      # 微信公众号:CV缝合救星
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)                                                                                                                      # 微信公众号:CV缝合救星

    def forward(self, x):
        b, c, h, w = x.size()

        # Adaptive pooling along H and W
        x_h = F.adaptive_avg_pool2d(x, (h, 1))  # H×1
        x_w = F.adaptive_avg_pool2d(x, (1, w))  # 1×W
        x_w = x_w.permute(0, 1, 3, 2)           # 调整维度以对齐

        # 拼接后输入共享 MLP
        y = torch.cat([x_h, x_w], dim=2)        # b×c×(h+w)×1
        y = self.shared_conv(y)

        # 分割为 H 和 W 两部分
        x_h_out, x_w_out = torch.split(y, [h, w], dim=2)
        x_w_out = x_w_out.permute(0, 1, 3, 2)   # 调整回原来维度

        # 分别映射到通道数
        a_h = self.conv_h(x_h_out*self.alpha).sigmoid()
        a_w = self.conv_w(x_w_out*self.alpha).sigmoid()

        # 融合：逐通道加权
        out = x * (a_h + a_w)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 32, 64, 64)  # batch=2, C=32, H=W=64                                                                                                                      # 微信公众号:CV缝合救星
    model = AdaptiveCoordAtt(in_channels=32, reduction=16)                                                                                                                      # 微信公众号:CV缝合救星
    print(model)
    print("\n微信公众号:CV缝合救星\n")
    y = model(x)
    print("输入:", x.shape)                                                                                                                                            # 微信公众号:CV缝合救星
    print("输出:", y.shape)                                                                                                                                             # 微信公众号:CV缝合救星
