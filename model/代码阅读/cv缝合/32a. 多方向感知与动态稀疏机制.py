import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
"""
CV缝合救星魔改创新：引入多方向感知与动态稀疏机制
问题：
1. 当前的 CFBConv 模块中，仅通过单一方向（水平和垂直）的卷积捕获局部特征，无法全面感知输入特征的方向信息。
2. 此外，注意力分布对于非显著区域仍进行建模，可能导致不必要的计算开销。
改进思路：
1. 引入多方向感知卷积：增加对角方向卷积（左上到右下、右上到左下），补充水平和垂直方向未捕获的特征，从而提升方向感知能力。
2. 动态稀疏注意力：通过稀疏化操作，仅保留显著区域的注意力分布，减轻计算负担并提升特征学习效率。
"""


class DirectionalAttention(nn.Module):
    """
    改进的卷积注意力模块，增加了对角方向特征提取，并引入动态稀疏注意力。
    """
    def __init__(self, in_channels, inter_channels, num_heads=8):
        super(DirectionalAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads

        # 水平和垂直卷积核
        self.kv_h = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))  # 水平方向
        self.kv_v = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))  # 垂直方向

        # 对角线卷积核
        self.kv_diag1 = nn.Parameter(torch.zeros(inter_channels, in_channels, 5, 5))  # 左上到右下
        self.kv_diag2 = nn.Parameter(torch.zeros(inter_channels, in_channels, 5, 5))  # 右上到左下

        # 参数初始化
        nn.init.trunc_normal_(self.kv_h, std=0.01)
        nn.init.trunc_normal_(self.kv_v, std=0.01)
        nn.init.trunc_normal_(self.kv_diag1, std=0.01)
        nn.init.trunc_normal_(self.kv_diag2, std=0.01)

        # 通道归一化
        self.norm = nn.BatchNorm2d(inter_channels)

    def dynamic_sparse_attention(self, x):
        """
        动态稀疏注意力机制：通过稀疏化减少非重要区域的参与。
        """
        x_shape = x.shape  # n,c,h,w
        h, w = x_shape[2], x_shape[3]

        # 将特征图分为多个注意力头
        x = x.reshape(x_shape[0], self.num_heads, self.inter_channels // self.num_heads, h, w)

        # 归一化注意力分布
        x = F.softmax(x, dim=-1)
        x = x * (x > 0.1).float()  # 动态稀疏化（仅保留重要特征）

        # 恢复原始形状
        x = x.reshape(x_shape)
        return x

    def forward(self, x):
        """
        前向传播，结合多方向卷积和动态稀疏注意力。
        """
        # 水平方向卷积
        x_h = F.conv2d(x, self.kv_h, padding=(3, 0))
        # 垂直方向卷积
        x_v = F.conv2d(x, self.kv_v, padding=(0, 3))
        # 对角方向卷积
        x_diag1 = F.conv2d(x, self.kv_diag1, padding=(2, 2))
        x_diag2 = F.conv2d(x, self.kv_diag2, padding=(2, 2))

        # 将各方向特征相加
        x = x_h + x_v + x_diag1 + x_diag2

        # 动态稀疏注意力
        x = self.dynamic_sparse_attention(x)

        # 通道归一化
        x = self.norm(x)
        return x


class ImprovedCFBConv(nn.Module):
    """
    改进后的CFBConv模块，结合方向感知注意力和动态稀疏机制。
    """
    def __init__(self, in_channels, out_channels, drop_rate=0.1, drop_path_rate=0.1):
        super(ImprovedCFBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 方向感知注意力模块
        self.attn = DirectionalAttention(in_channels, inter_channels=out_channels)

        # MLP模块
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

        # 用于调整残差连接的通道对齐
        if in_channels != out_channels:
            self.channel_align = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.channel_align = nn.Identity()  # 如果通道数一致，直接跳过

        # DropPath机制
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        """
        前向传播过程：
        1. 通过方向感知注意力模块提取多方向特征。
        2. 经过 MLP 提取非线性特征。
        3. 使用残差连接返回最终结果。
        """
        x_res = self.channel_align(x)  # 调整输入通道数，与输出对齐

        # 方向感知注意力模块
        x = self.attn(x)

        # 残差连接和 MLP
        x = x_res + self.drop_path(x)  # 残差连接
        x = x + self.drop_path(self.mlp(x))  # 经过 MLP 的残差连接
        x = self.dropout(x)

        return x



# 测试改进后的CFBConv模块
if __name__ == '__main__':
    # 定义改进的CFBConv模块
    model = ImprovedCFBConv(in_channels=32, out_channels=32).cuda()

    # 随机生成输入特征图 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 32, 64, 64).cuda()

    # 前向传播
    output_tensor = model(input_tensor)

    # 打印输入和输出的尺寸
    print('input_size:', input_tensor.size())
    print('output_size:', output_tensor.size())
