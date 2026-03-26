import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.layers import DropPath

'''
IDC大核分解卷积和INB模块在深度学习模型中的应用 (CVPR 2024)
即插即用模块： InceptionDWConv2d（IDC）和InceptionNeXtBlock（INB）

一、背景：近几年，受ViT等变换器结构在视觉任务上的优异表现的启发，研究人员开始采用大核卷积来扩展CNN的感受野，
以期获得类似于自注意力机制的效果。然而，使用大核卷积的模型在计算效率上表现欠佳。例如，ConvNeXt在7×7深度卷
积核的情况下尽管计算复杂度（FLOPs）较低，但内存访问成本却很高，导致在强计算设备上表现不佳。研究表明，减少卷
积核尺寸可以提高速度，但会显著降低性能，因此如何在保持性能的前提下提升大核卷积模型的效率成为一个重要挑战。

二、创新点:
1. 多分支大核分解卷积：IDC方法受Inception网络结构启发，将传统大核深度卷积分解为多个平行分支：
A. 小方形卷积核
B. 两个正交带状卷积核（宽和高方向的卷积核）
C.一个身份映射分支。通过这种方式
IDC模块能够有效增加感受野并提升计算效率。
2. 来提高计算效率：通过分解卷积核提升计算效率，减少大核卷积带来的高计算成本，实现速度与性能的平衡。
3. 扩大感受野：带状核能够在保持较低计算成本的情况下扩大感受野，从而捕捉更多的空间信息。
4. 性能优势：在不牺牲模型性能的前提下，InceptionNeXt 模块提高了推理速度，尤其适合高性能与高效率需求的场景。

三、适用任务：适用于图像分类、分割、目标检测等所有CV任务

'''
class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

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

class InceptionNeXtBlock(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
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

# 输入 B C H W   输出 B C H W
if __name__ == '__main__':
    # 创建输入张量
    input = torch.randn(1, 32, 64,64)

    INB = InceptionNeXtBlock(32)
    IDC = InceptionDWConv2d(32)

    output = INB(input)
    print('INB_input_size:',input.size())
    print('INB_output_size:',output.size())

    output = IDC(input)
    print('IDC_input_size:', input.size())
    print('IDC_output_size:', output.size())
