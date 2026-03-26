import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            # nn.Dropout(0.1)
        )

class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, rate=1):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            SeparableConvBNReLU(F1, F2, kernel_size, dilation=rate),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X
          
class MSAM(nn.Module):
    def __init__(self, dim_out):
        super(MSAM, self).__init__()
        self.branch1 = nn.Sequential(
            SeparableConvBNReLU(32, dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=1)
        )
        self.branch2 = nn.Sequential(
            SeparableConvBNReLU(64, dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=1)
        )
        self.branch3 = nn.Sequential(
            SeparableConvBNReLU(128, dim_out, kernel_size=3, stride=1)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(256, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU6()
        )
        self.merge = nn.Sequential(
            nn.Conv2d(4 * dim_out, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU6()
        )
        self.resblock = nn.Sequential(
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out]),
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out])
        )
        self.transformer = SelfAttention()
        self.conv = nn.Conv2d(1280, dim_out, 1)

    def forward(self, x_in4, x_in3, x_in2, x_in1):
        b, c, h, w = x_in4.shape
        list1 = []
        list2 = []

        x1 = self.branch1(x_in3)
        x2 = self.branch2(x_in2)
        x3 = self.branch3(x_in1)
        x0 = self.branch4(x_in4)

        # CNN
        merge = self.merge(torch.cat([x0, x1, x2, x3], dim=1))
        merge = self.resblock(merge)

        # Transformer
        list1.append(x0)
        list1.append(x3)
        list1.append(x2)
        list1.append(x1)

        for i in range(len(list1)):
            for j in range(len(list1)):
                if i <= j:
                    att = self.transformer(list1[i], list1[j])
                    list2.append(att)

        for j in range(len(list2)):
            list2[j] = list2[j].view(b, 128, h, w)

        out = self.conv(torch.concat(list2, dim=1))

        return out + merge


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.conv = SeparableConvBNReLU(256, 384, 3)

    def forward(self, x, y):
        b, c, h, w = x.shape
        fm = self.conv(torch.concat([x, y], dim=1))

        Q, K, V = rearrange(fm, 'b (qkv c) h w -> qkv b h c w'
                            , qkv=3, b=b, c=128, h=h, w=w)

        dots = (Q @ K.transpose(-2, -1))
        attn = dots.softmax(dim=-1)
        attn = attn @ V
        attn = attn.view(b, c, h, w)

        return attn


if __name__ == "__main__":

    dim_out = 128  # 输出的通道数

    # 构造输入张量 [B, C, H, W]
    
    x_in1 = torch.randn(1, 32, 256, 256)  # 输入张量 1
    x_in2 = torch.randn(1, 64, 128, 128)  # 输入张量 2
    x_in3 = torch.randn(1, 128, 64, 64)   # 输入张量 3
    x_in4 = torch.randn(1, 256, 64, 64)   # 输入张量 4

    # 实例化模型
    model = MSAM(dim_out=dim_out)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_in1 = x_in1.to(device)
    x_in2 = x_in2.to(device)
    x_in3 = x_in3.to(device)
    x_in4 = x_in4.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x_in4, x_in1, x_in2, x_in3)

    # 输出模型结构与形状信息
    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量 x_in1 形状:", x_in1.shape) 
    print("输入张量 x_in2 形状:", x_in2.shape) 
    print("输入张量 x_in3 形状:", x_in3.shape) 
    print("输入张量 x_in4 形状:", x_in4.shape)  
    print("输出张量形状       :", output.shape) 
