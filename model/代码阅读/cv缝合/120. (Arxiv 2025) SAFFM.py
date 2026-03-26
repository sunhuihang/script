import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpaCNN(nn.Module):
    def __init__(self, in_channels, SK_size = 3, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return F.relu(Y)

class SpatialCNNAtt(nn.Module):
    def __init__(self,in_channels = 64, SK_size = 3, kernel_size=3):
        super(SpatialCNNAtt, self).__init__()
        self.scnn = SpaCNN(in_channels=in_channels, SK_size=SK_size)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.scnn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SAFFM(nn.Module):
    def __init__(self, channel=64, SK_size = 3):
        super(SAFFM, self).__init__()
        self.spaCnnAtt = SpatialCNNAtt(in_channels=channel, SK_size=SK_size)
        self.chaAtt = ChannelAtt(in_channels=channel)
        self.conv1 = nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)
    def forward(self, x ,y):
        f = self.chaAtt(x) * self.spaCnnAtt(y)
        z = torch.cat([f*x, f*y], dim=1)
        return F.relu(self.bn1(self.conv1(z)))

if __name__ == "__main__":

    # 输入配置
    batch_size = 1
    channels = 32
    height = 256
    width = 256

    # 构造输入张量 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)

    # 实例化模型
    model = SAFFM(channel=channels, SK_size=3)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x, y)

    # 输出模型结构与形状信息
    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量 x 形状:", x.shape)      # [B, C, H, W]
    print("输入张量 y 形状:", y.shape)      # [B, C, H, W]
    print("输出张量形状   :", output.shape)    # [B, C, H, W]
