import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
try:
    from mmcv.cnn import ConvModule, build_norm_layer
    from mmengine.model import BaseModule
    from mmengine.model import constant_init
    from mmengine.model.weight_init import trunc_normal_init, normal_init
except ImportError as e:
    pass

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
    def forward(self, x):
        AiFHG = x
        conv1_x = self.conv1_AiFHG(x)
        conv2_x = self.conv2_AiFHG(x)
        conv3_x = self.conv3_AiFHG(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + AiFHG
        AiFHG = x
        x = self.projector(x)
        return AiFHG + x

class Mona(nn.Module):
    def __init__(self,in_dim,AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)
        self.dropout_AiFHG = nn.Dropout(p=0.1)
        self.adapter_conv_AiFHG = MonaOp(64)
        self.norm_AiFHG = nn.LayerNorm(in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, -1).transpose(-1, -2)
        AiFHG = x
        x = self.norm_AiFHG(x) * self.gamma_AiFHG + x * self.gammax_AiFHG
        project1 = self.project1_AiFHG(x)  #降维操作，减少计算量
        b, n, c = project1.shape
        h, w = H,W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)  #使用多尺度卷积操作，3*3，5*5，7*7代表不同大小的卷积核
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)   #使用激活函数
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)  #升维操作，还原方便残差连接
        out = AiFHG + project2
        out = out.reshape(B, H, W,C).permute(0,3,1,2)
        return out

if __name__ == "__main__":
    #创建Mona模块实例，32代表通道维度
    Mona = Mona(32)
    # 随机生成输入4维度张量：B, C, H, W
    input= torch.randn(1, 32,32,32)
    # 运行前向传递
    output = Mona(input)
    # 输出输入图片张量和输出图片张量的形状
    print(Mona)
    print("\n 哔哩哔哩：CV缝合救星!\n")
    print("CV_Mona_input size:", input.size())
    print("CV_Mona_Output size:", output.size())

