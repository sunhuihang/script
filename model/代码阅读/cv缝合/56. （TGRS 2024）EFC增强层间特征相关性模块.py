import math
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
   
56. 增强层间特征相关性（EFC）模块：优化特征融合提升小目标检测性能（TGRS 2024  SCI一区)
一、背景
1. 无人机图像小目标检测困境：在无人机图像里，小目标检测困难重重。小目标分辨率低，在深度网络中容易丢失特征，
而且极易受背景噪声干扰，有效信息少。传统基于特征金字塔网络（FPN）的多尺度特征融合方法，像简单的拼接或
相加操作，没有充分发挥多尺度融合的优势。这些方法在融合特征时，只是机械地堆叠和融合通道，没有考虑特征之
间的相关性，导致融合后出现冗余特征，无法很好地利用各层相关特征，限制了多尺度信息的表达，进而影响小目标
检测效果，在复杂背景和目标密集区域更是如此。
2. EFC 设计初衷：为应对这些问题并有效利用有限计算资源，EFC 模块被提出。其目的在于增强层间特征相关性，改进
特征融合策略，提高小目标检测性能，同时降低计算复杂度。

二、模块原理
1. 分组特征聚焦单元（GFF）
a. 空间聚焦与粗特征生成：先对来自不同阶段的低分辨率特征图进行上采样和通道调整，使其与高分辨率特征图在通道数
上一致，然后将两者相加得到粗特征。接着对粗特征进行处理，压缩为单通道并激活生成空间聚合权重，再用这个权重与粗
特征相乘，得到包含空间信息的特征。
b. 增强特征相关性：把包含空间信息的特征沿通道维度分组，在每组内对相邻通道特征进行交互操作，生成注意力掩码并应用于
精炼特征，最后把各组特征拼接起来，得到相关性高的相邻特征。
c. 空间映射归一化：将拼接后的特征嵌入多层原始特征融合的归一化层，用其均值和标准差进行归一化处理，从而获得具有强特
征相关性和丰富空间信息的特征。
2. 多级特征重建模块（MFR）
a. 特征分离：通过一系列操作由不同阶段特征得到一个综合特征，对其应用平均池化和激活函数生成信息权重阈值，同时分别处
理不同阶段特征得到各自权重信息，将这些权重信息与阈值比较，分离出强、弱特征。
b. 定向融合：把强特征的注意力图映射到综合特征上融合得到丰富特征，同样地，用弱特征注意力图处理得到弱特征。
c. 特征变换与融合：对丰富特征进行卷积操作使其信息更详细，对弱特征用专门设计的单元处理，最后把两者融合，得到既包含详
细信息又有跨通道信息交换的特征。

三、适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
'''
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class EFC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigomid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.one = c2
        self.two = c2
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1, )

    def forward(self, x1, x2):

        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigomid(bn_x)
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigomid(bn_x2)
        X_GOBAL = global_conv1 + global_conv2
        x_conv4 = self.conv4_gobal(X_GOBAL)
        X_4_sigmoid = self.sigomid(x_conv4)
        X_ = X_4_sigmoid * X_GOBAL
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=1)
            x1 = x_2_2.reshape(N, C, H, W)
            x1 = X_[group_id] * x1
            out.append(x1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)
        N, C, H, W = out.size()
        x_add_1 = out.reshape(N, self.group_num, -1)
        N, C, H, W = X_GOBAL.size()
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta)

        weight_x3 = self.Apt(X_GOBAL)
        reweights = self.sigomid(weight_x3)
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2
        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_genator(x_low)
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw
        xL = xL + x_gui
        return xL

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 64, 64, 64)
    # 初始化EFC模块并设定通道维度
    EFC_module = EFC(c1=32,c2=64) #c1表示input1通道数，c2表示input2通道数，
    output =EFC_module(input1,input2)#进行前向传播，输出通道数是C2
    # 输出结果的形状
    print("EFC_输入张量的形状：", input2.shape)
    print("EFC_输出张量的形状：", output.shape)
