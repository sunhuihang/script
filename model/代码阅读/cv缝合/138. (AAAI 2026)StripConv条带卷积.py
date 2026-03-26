import torch
import torch.nn as nn
# CV缝合救星
class StripConv(nn.Module):
    def __init__(self, dim, k1, k2): #K1=1 ,K2=19
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn

class StripModule(nn.Module):
    def __init__(self, d_model,k1=1,k2=19):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripConv(d_model,k1,k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)
    StripConv= StripModule(d_model=64,k1=1,k2=19)
    output =  StripConv(input)
    print('CV缝合救星即插即用模块永久更新-StripConv input_size:', input.size())
    print('CV缝合救星即插即用模块永久更新-StripConv output_size:', output.size())
    print('有关StripConv的二次创新，会更新在顶会顶刊二次创新改进交流群！')
    # MLKSA是AAAI2026 StripConv的二次创新模块,二次创新改进交流群会持续更新中ing
    # 二次创新群文件里面都是顶会顶刊论文模块的二次创新改进模块，可以直接发论文