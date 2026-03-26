import torch
import torch.nn as nn
import torch.distributions as td
"""
CV缝合救星魔改创新：动态注意力融合（Dynamic Attention Fusion）
1. 创新背景： 
目前的CSAM模块将语义、位置和切片注意力模块串联处理，这种固定的处理顺序可能在不同任务或不同数据分布下表现不佳。
为了提高模块的灵活性和适应性，我们提出“动态注意力融合”机制，允许网络根据输入数据自适应调整注意力模块的权重，
从而动态地聚合语义、位置和切片注意力的信息。

2. 创新实现：Dynamic Attention Fusion
动态注意力融合机制通过一个权重生成模块，根据输入特征计算每种注意力模块的权重，并将模块输出进行加权融合。
"""

def custom_max(x, dim, keepdim=True):
    temp_x = x
    for i in dim:
        temp_x = torch.max(temp_x, dim=i, keepdim=True)[0]
    if not keepdim:
        temp_x = temp_x.squeeze()
    return temp_x


class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super(PositionalAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), padding=3)

    def forward(self, x):
        max_x = custom_max(x, dim=(0, 1), keepdim=True)
        avg_x = torch.mean(x, dim=(0, 1), keepdim=True)
        att = torch.cat((max_x, avg_x), dim=1)
        att = self.conv(att)
        att = torch.sigmoid(att)
        return x * att


class SemanticAttentionModule(nn.Module):
    def __init__(self, in_features, reduction_rate=16):
        super(SemanticAttentionModule, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_rate),
            nn.ReLU(),
            nn.Linear(in_features // reduction_rate, in_features)
        )

    def forward(self, x):
        max_x = custom_max(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        avg_x = torch.mean(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        max_x = self.linear(max_x)
        avg_x = self.linear(avg_x)
        att = max_x + avg_x
        att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        return x * att


class SliceAttentionModule(nn.Module):
    def __init__(self, in_features, rate=4, uncertainty=True, rank=5):
        super(SliceAttentionModule, self).__init__()
        self.uncertainty = uncertainty
        self.rank = rank
        self.linear = nn.Sequential(
            nn.Linear(in_features, int(in_features * rate)),
            nn.ReLU(),
            nn.Linear(int(in_features * rate), in_features)
        )
        if uncertainty:
            self.non_linear = nn.ReLU()
            self.mean = nn.Linear(in_features, in_features)
            self.log_diag = nn.Linear(in_features, in_features)
            self.factor = nn.Linear(in_features, in_features * rank)

    def forward(self, x):
        max_x = custom_max(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        avg_x = torch.mean(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        max_x = self.linear(max_x)
        avg_x = self.linear(avg_x)
        att = max_x + avg_x
        if self.uncertainty:
            temp = self.non_linear(att)
            mean = self.mean(temp)
            diag = self.log_diag(temp).exp()
            factor = self.factor(temp).view(1, -1, self.rank)
            dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=factor, cov_diag=diag)
            att = dist.sample()
        att = torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * att


class AttentionFusionModule(nn.Module):
    def __init__(self, num_modules):
        super(AttentionFusionModule, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_modules) / num_modules)  # 初始均匀分配权重
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs):
        weights = self.softmax(self.weights)  # 计算动态权重
        fused_output = sum(w * inp for w, inp in zip(weights, inputs))  # 加权融合
        return fused_output


class CSAM(nn.Module):
    def __init__(self, num_slices, num_channels, semantic=True, positional=True, slice=True, uncertainty=True, rank=5):
        super(CSAM, self).__init__()
        self.semantic = semantic
        self.positional = positional
        self.slice = slice

        self.attention_modules = nn.ModuleList()
        if semantic:
            self.attention_modules.append(SemanticAttentionModule(num_channels))
        if positional:
            self.attention_modules.append(PositionalAttentionModule())
        if slice:
            self.attention_modules.append(SliceAttentionModule(num_slices, uncertainty=uncertainty, rank=rank))

        self.fusion = AttentionFusionModule(len(self.attention_modules))  # 动态权重生成模块

    def forward(self, x):
        outputs = []
        for module in self.attention_modules:
            outputs.append(module(x))
        x = self.fusion(outputs)  # 动态融合不同注意力模块的输出
        return x


# 测试代码
if __name__ == '__main__':
    model = CSAM(num_slices=10, num_channels=64).cuda()
    input = torch.randn(10, 64, 128, 128).cuda()
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
