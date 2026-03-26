import torch
import torch.nn as nn
import torch.nn.functional as F

class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1
    
class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c , 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x
# Dual-Stream Attention Module
class DSAM(nn.Module):
    def  __init__(self, in_c):
        super(DSAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = Pred_Layer(256 * 8)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return enhanced_feat, new_pred

if __name__ == "__main__":   

    # 定义输入张量的尺寸 (batch_size, channels, height, width)
    batch_size = 4
    channels = 256
    height = 64
    width = 64

    # 创建随机输入张量 feat 和 pred
    feat = torch.randn(batch_size, channels, height, width)
    pred = torch.randn(batch_size, 1, height, width)

    # 定义DSAM模型，输入通道数为256
    model = DSAM(in_c=channels)
    print(model)
    print("\n哔哩哔哩：CV缝合救星!\n")


    # 将输入张量 feat 和 pred 传入模型进行测试
    enhanced_feat, new_pred = model(feat, pred)

    # 打印输出张量的尺寸
    print(f'输入1 feat          : {feat.size()}')
    print(f'输入2 pred          : {pred.size()}')
    print(f'输出1 enhanced_feat : {enhanced_feat.size()}')
    print(f'输出2 new_pred      : {new_pred.size()}')