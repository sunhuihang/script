import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple
"""
矩形自校准模块（RCM）：用于高效语义分割的空间特征重建模块（ECCV 2024）
即插即用模块：RCM（可应用于多种网络，提升前景对象定位能力和特征表示）（添花模块）
一、背景
在语义分割任务中，尽管深度学习方法取得显著成果，但在有限计算资源下实现高级分割性能仍是挑战。
现有轻量级模型受特征表示能力限制，在建模前景对象边界和区分类别时存在困难，导致边界分割不准
确和分类错误。为解决这些问题，本文提出 RCM 模块，旨在提高模型对前景对象的定位能力和特征表
示。
二、RCM模块原理
1. 输入特征：来自网络的特征，其分辨率在不同层级有所不同，如在特定模型中，编码器产生的特征
分辨率。
2. 模块组成与处理：
A. 矩形自校准注意力（RCA）：采用水平池化和垂直池化捕获轴向全局上下文，生成两个轴向量，通过
广播加法建模矩形感兴趣区域。然后，利用两个大核条卷积设计形状自校准函数，分别在水平和垂直方
向校准注意力区域，使其更接近前景对象，且卷积权重可学习。
B. 特征融合：使用 3×3 深度卷积提取输入特征的局部细节，将校准后的注意力特征通过哈达玛积加权
到细化后的输入特征上。
C. 与 MetaNeXt 结构结合：在矩形自校准注意力后添加批归一化和 MLP 以细化特征，并采用残差连
接增强特征重用。
3. 输出：经过 RCM 处理后的特征，用于后续的空间特征重建和金字塔上下文提取。
三、适用任务
1. 该模块适用于语义分割任务，尤其在需要高效计算和准确分割前景对象的场景中表现出色。
2. 作为添花模块适用于所有CV任务。
"""

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather)  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc

        return out


class RCM(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=RCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size,
                                       ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64) #随机生成一张输入图片张量
    # 初始化RCM模块并设定通道维度
    rcm  = RCM(dim=32)
    output = rcm(input)  # 进行前向传播
    # 输出结果的形状
    print("RCM_输入张量的形状：", input.shape)
    print("RCM_输出张量的形状：", output.shape)