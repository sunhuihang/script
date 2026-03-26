import torch
import torch.nn as nn
import torch.nn.functional as F
"""
缺点2：下采样尺度固定
在现有的实现中，EASA分支的下采样尺度是固定的（self.down_scale = 8）。然而，对于不同分辨率的图像，固定的下
采样尺度可能会导致非局部特征捕获不足或过度平滑的现象。

CV缝合救星改进思路:
使用动态的下采样尺度，根据输入图像的分辨率或复杂度来调整下采样的程度。可以通过计算图像复杂度（例如图像的均方差）
来调整 down_scale，使其在高复杂度图像上使用较低的下采样比率，以保留更多信息。
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


class SMFADynamicDownscale(nn.Module):
    def __init__(self, dim=36, base_scale=8):
        super(SMFADynamicDownscale, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.base_scale = base_scale

    def forward(self, f):
        batch_size, channels, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)

        # 计算每个通道的标准差并取平均值
        std = torch.std(f, dim=(-2, -1), keepdim=True).mean()

        # 这里通过缩放和截断操作来近似整数下采样尺度
        down_scale_factor = torch.clamp(std * self.base_scale, min=1, max=self.base_scale)
        down_scale_h = torch.clamp(torch.floor(h / down_scale_factor).long(), min=1)
        down_scale_w = torch.clamp(torch.floor(w / down_scale_factor).long(), min=1)

        # 由于自适应池化要求输入为整数类型的尺寸
        x_down = F.interpolate(x, size=(down_scale_h, down_scale_w), mode='bilinear', align_corners=False)
        x_s = self.dw_conv(x_down)
        x_s_up = F.interpolate(x_s, size=(h, w), mode='bilinear', align_corners=False)

        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s_up * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='bilinear', align_corners=False)
        y_d = self.lde(y)

        return self.linear_2(x_l + y_d)


# 测试代码
if __name__ == '__main__':
    input_tensor = torch.rand(1, 32, 256, 256, requires_grad=True)
    model = SMFADynamicDownscale(dim=32)
    output = model(input_tensor)
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())

    # 简单测试反向传播
    loss = output.sum()
    loss.backward()
    print("Gradients are calculated successfully.")
