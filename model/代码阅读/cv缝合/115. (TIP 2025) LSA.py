import torch
import torch.nn as nn

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)

class LSA(nn.Module):
    def __init__(self, msfa_size, channel, reduction=16):
        super(LSA, self).__init__()
        self.compress = ChannelPool()
        self.shuffledown = Shuffle_d(msfa_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(msfa_size**2, msfa_size**2 // reduction, bias=False),
            nn.ReLU(inplace=True), # 微信公众号:CV缝合救星
            nn.Linear(msfa_size**2 // reduction, msfa_size**2, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(msfa_size)

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True), # 微信公众号:CV缝合救星
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        buff_x = x
        N, C, H, W = x.size()
        x = x.view(N * C, 1, H, W)  # N, C, H, W to N*C, 1, H, W
        sq_x = self.shuffledown(x)  # N*C, 1, H, W to N*C, 16, H/4, W/4
        b, c, _, _ = sq_x.size() # 微信公众号:CV缝合救星
        y = self.avg_pool(sq_x).view(b, c)  # N*C, 16, H/4, W/4 to N*C, 16, 1, 1 to N*C, 16
        y = self.fc(y).view(b, c, 1, 1)  # N*C, 16, 1, 1
        y = y.expand_as(sq_x)  # N*C, 16, 1, 1 to N*C, 16, H/4, W/4
        ex_y = self.shuffleup(y)  # N*C, 16, H/4, W/4 to N*C, 1, H, W
        out = x * ex_y  # 微信公众号:CV缝合救星
        out = out.view(N, C, H, W)

        b, c, _, _ = buff_x.size()
        y = self.avg_pool1(buff_x).view(b, c)
        y = self.fc1(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)
        return out


class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)


if __name__ == "__main__":

    batch_size = 1
    channels = 32         # 输入通道数
    height = 256
    width = 256
    msfa_size = 4         # 用于 PixelShuffle 和 ChannelShuffle 的缩放因子

    # 构造输入张量
    input_tensor = torch.randn(batch_size, channels, height, width)

    # 模型实例化
    model = LSA(msfa_size=msfa_size, channel=channels)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # 前向传播
    output = model(input_tensor)

    # 输出模型结构与输入输出形状
    print(model)
    print("\n哔哩哔哩:CV缝合救星\n")
    print("输入张量形状:", input_tensor.shape)   # [B, C, H, W]
    print("输出张量形状:", output.shape)        # [B, C, H, W]
