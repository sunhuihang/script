import torch.nn as nn
import torch
import torch.nn.functional as F


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))

        self.conv_1x1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()

        self.F_h = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_w = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        x_1, x_2 = x.split(C, 1)

        x_1 = self.F_h(x_1)
        x_2 = self.F_w(x_2)
        s_h = self.sigmoid(x_1)
        s_w = self.sigmoid(x_2)

        out = res * s_h.expand_as(res) * s_w.expand_as(res)

        return out


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CrossScaleContextModule(nn.Module):
    def __init__(self, n_feat):
        super(CrossScaleContextModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_feat * 3, n_feat // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_feat // 16, n_feat * 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        b, c, _, _ = x1.size()
        y1 = self.avg_pool(x1).view(b, c)
        y2 = self.avg_pool(x2).view(b, c)
        y3 = self.avg_pool(x3).view(b, c)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.fc(y).view(b, c * 3, 1, 1)
        scale1, scale2, scale3 = torch.split(y, c, dim=1)
        out1 = x1 * scale1.expand_as(x1)
        out2 = x2 * scale2.expand_as(x2)
        out3 = x3 * scale3.expand_as(x3)
        return out1, out2, out3


class RCSSC(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        self.SA = spatial_attn_layer()
        self.CA = CALayer(n_feat, reduction)

        # 多尺度卷积层
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(n_feat, n_feat, kernel_size=7, padding=3, stride=1)
        ])

        # 跨尺度上下文感知模块
        self.cross_scale_context = CrossScaleContextModule(n_feat)

        # 根据实际拼接通道数修改conv1x1的输入通道数
        total_channels = len(self.multi_scale_conv) * n_feat + n_feat * 2
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(total_channels, 32, kernel_size=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)

        # 多尺度特征提取
        multi_scale_features = [conv(x) for conv in self.multi_scale_conv]

        # 跨尺度上下文感知融合
        multi_scale_features[0], multi_scale_features[1], multi_scale_features[2] = self.cross_scale_context(
            multi_scale_features[0], multi_scale_features[1], multi_scale_features[2])

        multi_scale_features.append(sa_branch)
        multi_scale_features.append(ca_branch)

        # 特征拼接
        x1 = torch.cat(multi_scale_features, dim=1)

        x1 = self.conv1x1(x1)
        x2 = torch.sigmoid(
            torch.add(x, F.interpolate(self.SC(x), x.size()[2:])))
        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out


if __name__ == '__main__':
    batch_size = 1
    dim = 32
    height, width = 256, 256

    x = torch.randn(batch_size, dim, height, width)

    model = RCSSC(n_feat=dim, reduction=16)

    print(model)
    print("哔哩哔哩: CV缝合救星!")

    output = model(x)

    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")
