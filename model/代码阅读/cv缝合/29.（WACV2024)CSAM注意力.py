import torch
import torch.nn as nn
import torch.distributions as td
'''
CSAM: 2.5D Cross-Slice Attention Module for Anisotropic Volumetric Medical Image 
Segmentation (WACV 2024)
即插即用模块： CSAM跨切片注意力模块（强化模块）

一、背景
医学影像中的大部分体数据（如MRI）存在各向异性，其横向分辨率通常远低于平面内分辨率。这种分辨率差异对
传统2D或3D分割方法提出了挑战：
1. 2D方法：忽略切片间的三维信息。
2. 3D方法：在各向异性数据下表现不佳，且对体积重采样需求高。
当前2.5D方法试图结合2D卷积和体积信息，通过切片堆叠实现跨切片关系建模，但常因参数量庞大而难以优化。
为此，CSAM模块被提出，显著减少可训练参数，通过语义、位置和切片注意力实现全局跨切片信息提取，并提升
分割性能。

二、CSAM模块原理
1. 输入特征：接受整个体数据的深度特征图（多尺度输入）。
2. 注意力机制：
A. 语义注意力：通过最大池化和平均池化结合多层感知机（MLP）捕获全体数据的语义重要性。
B. 位置注意力：对每个切片的空间信息进行建模，定位关键区域。
C. 切片注意力：根据切片间的任务相关性加权分配重要性，使用高斯分布捕获不确定性以规避假阳性。
3. 输出特征：整合上述注意力模块后的加权特征图。

三、适用任务
1. 医学图像分割：特别适用于各向异性MRI数据（前列腺、胎盘、心脏等分割任务）。
在基础网络（如U-Net及其变种）中插入CSAM显著提高分割性能。
2. 跨领域推广：可应用于其他需要跨维度建模的计算机视觉任务。
3. 资源优化：相较其他2.5D方法，显著降低参数量，适合资源受限环境。

'''

def custom_max(x,dim,keepdim=True):
    temp_x=x
    for i in dim:
        temp_x=torch.max(temp_x,dim=i,keepdim=True)[0]
    if not keepdim:
        temp_x=temp_x.squeeze()
    return temp_x

class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super(PositionalAttentionModule,self).__init__()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(7,7),padding=3)
    def forward(self,x):
        max_x=custom_max(x,dim=(0,1),keepdim=True)
        avg_x=torch.mean(x,dim=(0,1),keepdim=True)
        att=torch.cat((max_x,avg_x),dim=1)
        att=self.conv(att)
        att=torch.sigmoid(att)
        return x*att

class SemanticAttentionModule(nn.Module):
    def __init__(self,in_features,reduction_rate=16):
        super(SemanticAttentionModule,self).__init__()
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=in_features//reduction_rate))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=in_features//reduction_rate,out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
    def forward(self,x):
        max_x=custom_max(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        avg_x=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        att=torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        return x*att

class SliceAttentionModule(nn.Module):
    def __init__(self,in_features,rate=4,uncertainty=True,rank=5):
        super(SliceAttentionModule,self).__init__()
        self.uncertainty=uncertainty
        self.rank=rank
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=int(in_features*rate)))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=int(in_features*rate),out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
        if uncertainty:
            self.non_linear=nn.ReLU()
            self.mean=nn.Linear(in_features=in_features,out_features=in_features)
            self.log_diag=nn.Linear(in_features=in_features,out_features=in_features)
            self.factor=nn.Linear(in_features=in_features,out_features=in_features*rank)
    def forward(self,x):
        max_x=custom_max(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        avg_x=torch.mean(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        if self.uncertainty:
            temp=self.non_linear(att)
            mean=self.mean(temp)
            diag=self.log_diag(temp).exp()
            factor=self.factor(temp)
            factor=factor.view(1,-1,self.rank)
            dist=td.LowRankMultivariateNormal(loc=mean,cov_factor=factor,cov_diag=diag)
            att=dist.sample()
        att=torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x*att


class CSAM(nn.Module):
    def __init__(self,num_slices,num_channels,semantic=True,positional=True,slice=True,uncertainty=True,rank=5):
        super(CSAM,self).__init__()
        self.semantic=semantic
        self.positional=positional
        self.slice=slice
        if semantic:
            self.semantic_att=SemanticAttentionModule(num_channels)
        if positional:
            self.positional_att=PositionalAttentionModule()
        if slice:
            self.slice_att=SliceAttentionModule(num_slices,uncertainty=uncertainty,rank=rank)
    def forward(self,x):
        if self.semantic:
            x=self.semantic_att(x)
        if self.positional:
            x=self.positional_att(x)
        if self.slice:
            x=self.slice_att(x)
        return x
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    models = CSAM(num_slices=10,num_channels=64).cuda()
    input = torch.randn(10, 64, 128, 128).cuda()
    output = models(input)
    print('input_size:',input.size())
    print('output_size:',output.size())