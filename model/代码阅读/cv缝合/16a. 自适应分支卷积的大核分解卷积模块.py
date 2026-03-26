import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CV缝合救星魔改1: 自适应分支卷积的大核分解卷积模块

一、缺点：在标准卷积模块中，所有通道都参与相同的卷积操作，缺乏适应性和灵活性，计算负担较重，尤其在不同类型特征混合时效果欠佳。
二、CV缝合救星魔改：引入自适应分支卷积，通过自适应分支分配和轻量化卷积模块减少计算成本，并增强模型的感受野。
    1. 使用多分支分解：将输入通道分配给不同分支，分别进行身份映射、方形卷积、宽带卷积和高带卷积。
    2. 融合层：使用1x1卷积将多分支输出融合，进一步提取并合并特征。
    3. 通道注意力机制：利用SE模块赋予各卷积分支自适应的权重，以更灵活地关注不同特征。
"""


class SEBlock(nn.Module):
    """ 通道注意力（SE模块） """

    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)

    def forward(self, x):
        # 全局平均池化，获得通道的全局特征
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class AdaptiveInceptionDWConv2d(nn.Module):
    """ 自适应分支卷积 + 通道注意力的大核分解卷积模块 """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=7, branch_ratio=0.125):
        """
        Args:
            in_channels (int): 输入通道数
            square_kernel_size (int): 方形卷积核大小（默认值为3）
            band_kernel_size (int): 带状卷积核大小（默认值为7）
            branch_ratio (float): 分配给每个卷积分支的通道比例（默认值为0.125）
        """
        super().__init__()

        self.gc = int(in_channels * branch_ratio)  # 每个卷积分支的通道数

        # 定义方形卷积分支，并添加SE模块
        self.dwconv_hw = nn.Sequential(
            nn.Conv2d(self.gc, self.gc, square_kernel_size, padding=square_kernel_size // 2, groups=self.gc),
            SEBlock(self.gc)
        )

        # 定义宽方向的带状卷积分支，并添加SE模块
        self.dwconv_w = nn.Sequential(
            nn.Conv2d(self.gc, self.gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                      groups=self.gc),
            SEBlock(self.gc)
        )

        # 定义高方向的带状卷积分支，并添加SE模块
        self.dwconv_h = nn.Sequential(
            nn.Conv2d(self.gc, self.gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                      groups=self.gc),
            SEBlock(self.gc)
        )

        # 剩余通道作为身份映射
        self.identity_channels = in_channels - 3 * self.gc

        # 使用1x1卷积将分支输出进行融合
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=1)

    def forward(self, x):
        # 将输入张量按通道数切分为身份映射分支、方形卷积分支、宽带卷积分支和高带卷积分支
        x_id, x_hw, x_w, x_h = torch.split(x, [self.identity_channels, self.gc, self.gc, self.gc], dim=1)

        # 各卷积分支的输出
        out_hw = self.dwconv_hw(x_hw)
        out_w = self.dwconv_w(x_w)
        out_h = self.dwconv_h(x_h)

        # 拼接所有分支的输出
        out = torch.cat((x_id, out_hw, out_w, out_h), dim=1)

        # 通过1x1卷积进行融合
        out = self.fusion(out)

        return out


# 测试模块
if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 64, 64)  # 创建一个示例输入张量
    model = AdaptiveInceptionDWConv2d(in_channels=32)  # 创建自适应分支卷积的 IDC 模块
    output = model(input_tensor)
    print('输入大小:', input_tensor.size())
    print('输出大小:', output.size())
