import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

"""
DySample：高效动态上采样器（ICCV 2023）
即插即用模块：DySample

一、背景
在密集预测任务中（如语义分割、目标检测等），特征的逐步上采样是模型恢复特征分辨率的关键过程。传统的上采样方法，
如双线性插值和最近邻插值，计算开销相对较小，但灵活性不足，无法针对具体特征内容进行优化。而一些动态上采样器（
如CARAFE、FADE、SAPA）虽然有很好的性能，但往往需要大量的计算资源，且推理速度较慢。因此，为实现计算效率与效
果之间的平衡，提出了DySample，一种超轻量且高效的动态上采样器，尤其适用于计算资源受限的环境。

二、DySample的原理
点采样重新定义上采样：DySample从点采样的角度重新定义上采样过程，绕过了传统的动态卷积。具体来说，它通过生成内
容感知的采样点，对输入特征图进行重采样，来实现上采样的效果。这个过程使用PyTorch的内置函数grid_sample，大大
简化了实现难度和计算开销。
模块机制：
A. 初始采样位置优化：DySample提出了从双线性插值初始化开始的方法，通过合理设定初始采样位置，确保采样点在上采样
后能更均匀地分布，从而提升采样精度。
B. 控制偏移范围：为了减少采样点重叠可能引入的边界预测错误，DySample引入了范围因子，控制偏移范围，使采样过程更
加稳定。
C. 分组上采样：DySample通过将特征通道分组并为每个组生成独立的采样集，进一步提升了上采样的灵活性和性能。
D. 轻量化与高效性：DySample通过简单的偏移生成和内置的点采样函数，减少了对自定义CUDA包的依赖，降低了FLOPs、显
存和推理延迟。它在多个任务中表现出色，并且其推理速度接近于双线性插值，而性能远超传统方法。

三、适用任务
DySample适用于各种密集预测任务，包括语义分割、目标检测、实例分割、全景分割和单目深度估计等，特别是在计算资源
有限的设备（如边缘设备和移动设备）上表现优异。DySample无需高分辨率引导特征输入，且易于实现，适合替代现有的最
近邻或双线性插值，显著提高模型的性能与效率。
"""

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample_UP(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super(DySample_UP,self).__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("lp",i.shape)
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("pl",i.shape)
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
    '''
   1. 'lp'（局部感知）：这种风格直接在输入特征图的每个局部区域生成偏移量，然后基于这些偏移进行上采样。这意味着每个输出像素的
   位置都直接受到对应输入区域内容的影响，适用于需要精细控制每个输出位置如何从输入特征中取样的情况。在需要保持局部特征连续
   性和细节信息的任务（如图像超分辨率、细节增强）中，'lp' 风格可能表现更佳。
   2. 'pl'（像素shuffle后局部感知）：在应用偏移量之前，首先通过像素shuffle操作重新排列输入特征图的像素，这实际上是一种空间
   重排，能够促进通道间的信息交互。随后，再进行与'lp'类似的局部感知上采样。这种风格有助于增强全局上下文信息的融合和特征的
   重新组织，适合需要依赖相邻区域上下文信息的任务（例如语义分割、全景分割）。像素shuffle增加了特征图的表征能力，帮助模型
   捕捉更广泛的上下文信息。
   3. 两者的优势比较：如果任务更强调保留和增强局部细节，那么 'lp' 风格可能是更好的选择；而如果任务需要更多的全局上下文信息
   和特征重组，'pl' 风格可能更合适。
    '''

if __name__ == '__main__':
    input = torch.rand(1, 64, 4, 4)
    # in_channels=64, scale=4, style='lp'/‘pl’,
    DySample_UP = DySample_UP(in_channels=64,scale=2,style='pl')
    output = DySample_UP(input)
    print('input_size:', input.size())
    print('output_size:', output.size())