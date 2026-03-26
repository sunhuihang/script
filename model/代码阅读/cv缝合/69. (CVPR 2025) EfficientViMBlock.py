import torch
import torch.nn as nn
import math

# B站:CV缝合救星
"""
69. EfficientViM：基于隐状态混合器的状态空间对偶高效视觉 Mamba 模块 (2025 CVPR)
替身强化模块-》Vit ViM

一、背景
在资源受限环境下部署神经网络时，传统卷积神经网络（CNN）和基于注意力机制
的视觉 Transformer（ViT）在捕捉图像局部和全局依赖关系方面各有优劣。CNN 中的深
度可分离卷积（DWConv）虽能构建轻量级架构，但自注意力机制的二次复杂度限制了 ViT
在处理高分辨率图像时的效率。近期，状态空间模型（SSM）以其线性计算复杂度在全局令牌
交互中崭露头角，然而基于 SSM 构建的高效视觉骨干网络仍有待深入探索。为解决这些问题，
提出 EfficientViM 模块，旨在实现高效的全局依赖捕捉，同时优化计算成本和模型性能。

二、EfficientViM 模块介绍
（一）整体设计EfficientViM 是一种基于隐状态混合器的状态空间对偶（HSM-SSD）
构建的新型轻量级视觉骨干网络架构。通过重新设计状态空间对偶层，将通道混合操作转移到隐
状态空间，降低计算成本。同时，引入多阶段隐状态融合机制，结合原始 logits 和各阶段隐状
态衍生的 logits 进行模型预测，增强隐状态的表示能力。此外，通过优化设计减少内存受限操作，
优先考虑实际应用中的性能。
（二）核心组件与操作
1. 隐状态混合器的状态空间对偶（HSM-SSD）：重新设计 NC-SSD 层，将计算量大的投影操
作转移到隐状态空间，通过调整隐状态数量降低计算复杂度。提出隐状态混合器（HSM），在压
缩后的隐状态数组上直接进行通道混合、门控和输出投影操作，进一步降低计算成本。
2. 多阶段隐状态融合（MSF）：融合网络多个阶段的隐状态来生成预测 logits。对每个阶段最
后一个模块的隐状态进行全局平均池化得到全局表示，再将其归一化和投影生成对应 logits，最
终的 logit 是所有阶段 logits 的加权和，强化了隐状态的表示能力，提升模型泛化能力。单头
HSM-SSD 设计：针对多头设计在 HSM-SSD 中带来的内存瓶颈问题，采用单头设计并设置状态相关
的重要性权重，在保持性能的同时提高了吞吐量。

三、微观设计考量
HSM-SSD 层通过调整隐状态数量和在隐状态空间进行操作，有效降低了计算复杂度。多阶段隐状态
融合机制充分利用了不同阶段的特征信息，增强了模型的表示能力。单头设计减少了内存受限操作，
提高了运行效率。消融实验验证了HSM-SSD、单头设计和多阶段融合等组件的有效性，表明 
EfficientViM 在速度 - 精度权衡方面优于其他模型。

四、适用任务
EfficientViM 适用于多种视觉任务，如 ImageNet-1K 图像分类、COCO 目标检测与实例分割等。
在图像分类任务中，EfficientViM 在速度和精度上均超越了先前的高效网络，例如 EfficientViM-M2 
比 MobileViTV2 0.75 速度快约 4 倍，精度提高 0.2%。在高分辨率图像场景下，EfficientViM
相比其他模型具有更好的扩展性，随着分辨率提升，其与 SHViT 的速度差距增大，在 512² 分辨率
下比 SHViT 快 15% 以上。在使用蒸馏训练时，EfficientViM 也能在速度 - 精度权衡上取得优
异表现 。
"""

class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized

class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
    
    
class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)
        
    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim*3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0) 
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L= x.shape
        H = int(math.sqrt(L))
        
        BCdt = self.dw(self.BCdt_proj(x).view(batch,-1, H, H)).flatten(2)
        B,C,dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1) 
        A = (dt + self.A.view(1,-1,1)).softmax(-1) 
        
        AB = (A * B) 
        h = x @ AB.transpose(-2,-1) 
        
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1) 
        h = self.out_proj(h * self.act(z)+ h * self.D)
        y = h @ C # B C N, B C L -> B C L
        
        y = y.view(batch,-1,H,H).contiguous()# + x * self.D  # B C H W
        return y, h
    
class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand,state_dim=state_dim)  
        self.norm = LayerNorm1D(dim)
        
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        
        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))
        
        #LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4,dim), requires_grad=True)
        
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4,-1,1,1)
        
        # DWconv1
        x = (1-alpha[0]) * x + alpha[0] * self.dwconv1(x)
        
        # HSM-SSD
        x_prev = x
        x, h = self.mixer(self.norm(x.flatten(2))) 
        x = (1-alpha[1]) * x_prev + alpha[1] * x
        
        # DWConv2
        x = (1-alpha[2]) * x + alpha[2] * self.dwconv2(x)
        
        # FFN
        x = (1-alpha[3]) * x + alpha[3] * self.ffn(x)
        # return x, h
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 32, 256, 256).to(device)

    evim = EfficientViMBlock(dim=32).to(device)
    print(evim)

    output = evim(x)

    print("input_size", x.shape)
    print("output_size", output.shape)