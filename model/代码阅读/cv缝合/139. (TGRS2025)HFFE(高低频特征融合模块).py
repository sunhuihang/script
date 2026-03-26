import torch
import torch.nn as nn
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs                                                                                                           #来自Ai缝合怪复现整理
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size                                                                     #来自Ai缝合怪复现整理
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

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


class CoordAttiton(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
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
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()                                                                          #来自Ai缝合怪复现整理

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * res

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x_source
class HFFE(nn.Module):
    def __init__(self, feature_low_channel, feature_high_channel, out_channel, kernel_size=3):
        super(HFFE, self).__init__()
        self.conv_block_low = nn.Sequential(
            CBR(feature_low_channel, feature_low_channel // 16, kernel_size),
            nn.Conv2d(feature_low_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()                                                                                                                                                                                          ##来自Ai缝合怪复现整理
        )

        self.conv_block_high = nn.Sequential(
            CBR(feature_high_channel, feature_high_channel // 16, kernel_size),
            nn.Conv2d(feature_high_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = CBR(feature_low_channel, out_channel, 1)
        self.conv2 = CBR(feature_high_channel, out_channel, 1)
        self.conv3 = CBR(feature_low_channel + feature_high_channel, out_channel, 1)

        self.Up_to_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)                                                                                                       #来自Ai缝合怪复现整理

        self.feature_low_sa = SpatialAttention()
        self.feature_high_sa = SpatialAttention()

        self.ca = CoordAttiton(out_channel,out_channel)

        self.conv_final = CBR(out_channel * 2, out_channel, 1)

    def forward(self,x_low, x_high):
        b1, c1, w1, h1 = x_low.size()
        b2, c2, w2, h2 = x_high.size()
        if (w1, h2) != (w2, h2):
            x_high = self.Up_to_2(x_high)

        source_low = x_low
        source_high = x_high

        x_low = self.feature_low_sa(x_low)
        x_high = self.feature_high_sa(x_high)

        x_low_map = self.conv_block_low(x_low)
        x_high_map = self.conv_block_high(x_high)

        x_mix = torch.cat([source_low * x_high_map, source_high * x_low_map], 1)
        x_ca = torch.sigmoid(self.ca(self.conv3(x_mix)))

        x_low_att = x_ca * self.conv1((source_low + x_low))
        x_high_att = x_ca * self.conv2((source_high + x_high))
        out = self.conv_final(torch.cat([x_low_att, x_high_att], 1))

        return out

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 创建 HFFE 模块
    model = HFFE(feature_low_channel=32, feature_high_channel=32, out_channel=32)
    # 将输入图像传入 HFFE 模块进行处理
    output = model(input1,input2)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('CV缝合救星即插即用模块永久更新-HFFE_input_size:', input1.size())
    print('CV缝合救星即插即用模块永久更新-HFFE_output_size:', output.size())
    print('有关HFFE的二次创新，会更新在顶会顶刊二次创新改进交流群！')