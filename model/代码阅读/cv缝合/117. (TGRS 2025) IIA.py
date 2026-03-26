import torch
import torch.nn as nn


class AttentionWeight(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(AttentionWeight, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size, padding=padding, groups=channel, bias=False)
        self.bn = nn.BatchNorm1d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, w, c, h = x.size()
        x_weight = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_weight = self.conv1(x_weight).view(b, c, h)
        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        x_weight = x_weight.view(b, 1, c, h)

        return x * x_weight


class IIA(nn.Module):
    def __init__(self, channel):
        super(IIA, self).__init__()
        self.attention = AttentionWeight(channel)

    def forward(self, x):
        # b, w, c, h
        x_h = x.permute(0, 3, 1, 2).contiguous()
        x_h = self.attention(x_h).permute(0, 2, 3, 1)
        # b, h, c, w
        x_w = x.permute(0, 2, 1, 3).contiguous()
        x_w = self.attention(x_w).permute(0, 2, 1, 3)
        # b, c, h, w
        # x_c = self.attention(x)

        # return x + 1 / 2 * (x_h + x_w)  # 89.8	92.5	81.9
        return x + x_h + x_w


class ChannelAttention(nn.Module):
    def __init__(self, inp, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(inp, inp // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inp // ratio, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


if __name__ == "__main__":
    batch_size = 1
    channels = 32
    height = 256
    width = 256

    # 输入张量 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)

    # 实例化模型
    model = IIA(32)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x)

    # 输出模型结构与形状信息
    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]