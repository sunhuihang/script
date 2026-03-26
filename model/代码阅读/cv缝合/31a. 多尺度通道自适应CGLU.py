import torch
import torch.nn as nn
"""
CV缝合救星魔改创新1：引入多尺度特征融合
问题：当前的CGLU模块仅通过单尺度的3×3深度卷积捕获局部特征，可能对输入特征的多尺度信息感知不足。
改进思路：
1. 引入一个多尺度机制，例如同时添加不同卷积核尺寸（如3×3和5×5）的深度卷积。
2. 不同尺度的特征可以在深度卷积后通过拼接或加权融合实现增强的特征表达。

CV缝合救星魔改创新2：引入通道自适应机制
问题：当前的CGLU模块中，通道间的权重调整是隐式的（通过GLU中的门控分支实现），缺乏全局上下文信息的显式
建模能力。
改进思路：
添加一个轻量级的通道注意力机制（如Squeeze-and-Excitation, SE）到CGLU模块中，通过显式建模全局上下文，
对特征进行通道维度的自适应增强。
"""
# 多尺度深度卷积模块
class MultiScaleDWConv(nn.Module):
    def __init__(self, dim=768):
        super(MultiScaleDWConv, self).__init__()
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = self.dwconv3(x)
        x5 = self.dwconv5(x)
        x = (x3 + x5) / 2  # 或者改为加权融合
        x = x.flatten(2).transpose(1, 2)
        return x

# 通道自适应机制（SE模块）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入形状 [B, N, C]
        B, N, C = x.size()
        x_pool = self.global_avg_pool(x.transpose(1, 2)).view(B, C)  # 全局平均池化 [B, C]
        x_se = self.sigmoid(self.fc2(self.relu(self.fc1(x_pool)))).view(B, 1, C)  # 通道权重 [B, 1, C]
        return x * x_se  # 加权特征

# CGLU模块
class CGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(CGLU, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = MultiScaleDWConv(hidden_features)  # 替换为多尺度卷积
        self.se = SEBlock(hidden_features)  # 添加通道自适应模块
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v  # 门控机制
        x = self.se(x)  # 通道自适应增强
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 测试代码
if __name__ == '__main__':
    models = CGLU(in_features=768, hidden_features=512, out_features=768)

    # 1. 输入4维图片张量（CV方向）
    input_img = torch.randn(2, 768, 14, 14)  # 输入形状 [B, C, H, W]
    input = input_img
    input_img = input_img.reshape(2, 768, -1).transpose(-1, -2)  # 转为 [B, N, C]
    output = models(input_img, 14, 14)
    output = output.view(2, 768, 14, 14)  # 输出转回 [B, C, H, W]
    print("CV_CGLU_input size:", input.size())
    print("CV_CGLU_Output size:", output.size())

    # 2. 输入3维特征张量（NLP方向）
    B, N, C = 2, 196, 768  # 批量大小、序列长度、特征维度
    H, W = 14, 14  # 重塑后的高度和宽度
    input = torch.randn(B, N, C)  # 输入形状 [B, N, C]
    output = models(input, H, W)
    print('NLP_CGLU_input size:', input.size())
    print('NLP_CGLU_output size:', output.size())
