import torch
import torch.nn as nn
import torch.nn.functional as F

'''
MSA2Net: 多尺度自适应注意力引导的医学图像分割网络 (BMVC 2024)
即插即用模块：MASAG 

一、背景：文章探讨了医学图像分割的挑战和限制，尤其是在处理各种组织和结构时，尺寸、形状和密度的显著变化增加了识别的复杂性。
传统的卷积神经网络（CNN）在捕捉长距离依赖方面存在局限，而Transformer模型虽具备自注意力机制，能够处理全局信息，但在处理
局部细节和计算效率上存在不足。因此，本文提出了MSA2Net，一个新的深度分割框架，利用改进的跳跃连接和多尺度自适应空间注意门
（MASAG）来融合局部和全局特征，以提高医学图像分割的精确度和效率。

二、MASAG模块机制：
1. 输入：X 和 G：分别代表编码器（encoder）和解码器（decoder）的输入特征。通常，X 包含较高分辨率的空间信息，而 G 包
含丰富的语义信息。
2. 多尺度特征融合
A. 局部上下文提取：通过深度卷积（depthwise convolution）增强了X的空间范围。
B. 全局上下文提取：使用最大池化和平均池化从G中提取全局信息。这两种池化技术提取不同尺度的统计信息，帮助模型捕捉全局特征。
C. 结合这两部分的输出后，通过1x1卷积进行特征融合，生成一个新的特征图（U），为下一步的空间选择做准备。
3. 空间选择
A. 特征图U经过1x1卷积，映射到两个通道，每个通道的输出通过softmax进行归一化，生成空间选择权重。
B. 这些权重用于调整特征图X和G，通过加权和加上原始输入（残差连接），生成新的特征图X'和G'。
4. 空间交互与交叉调制
A. X'和G'通过sigmoid激活函数处理，使得每个特征图都能在保留其特定的空间信息的同时，也融合对方的信息。
B. 这种交互增强了局部细节与全局上下文之间的联系，生成最终的融合特征图U'。
5. 重新校准
A. U'经过点卷积和sigmoid激活，生成注意力图，这个注意力图用来重新校准初始输入X。
B. 经过点卷积调整后的X与注意力图相乘，以此来突出重要特征，并抑制不重要的背景信息。

三、适用任务：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块。
'''

def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)

    fuse = self.bn(x + g)
    return fuse


class MASAG(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))
  def forward(self,x,g):
    x_ = x.clone()
    g_ = g.clone()
    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W
    multi = self.multi(x, g) # B, C, H, W
    ### Option 2 ###
    multi = self.selection(multi) # B, num_path, H, W

    attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
    #attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_
    x_att = x_att + x_
    g_att = g_att + g_
    ## Bidirectional Interaction
    x_sig = torch.sigmoid(x_att)
    g_att_2 = x_sig * g_att
    g_sig = torch.sigmoid(g_att)
    x_att_2 = g_sig * x_att
    interaction = x_att_2 * g_att_2
    projected = torch.sigmoid(self.bn(self.proj(interaction)))
    weighted = projected * x_
    y = self.conv_block(weighted)
    #y = self.bn_2(weighted + y)
    y = self.bn_2(y)
    return y

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)

    # 创建一个MASAG实例
    MASAG = MASAG(dim=64)

    # 将两个输入特征图传递给 MSGA 模块
    output = MASAG(input1, input2)
    # 打印输入和输出的尺寸
    print(f"input 1 shape: {input1.shape}")
    print(f"input 2 shape: {input2.shape}")
    print(f"output shape: {output.shape}")