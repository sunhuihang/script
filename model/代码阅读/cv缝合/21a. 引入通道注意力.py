import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新1：引入通道注意力机制
特征提取的选择性不足：
原始的 Partial_conv3 模块虽然可以通过对部分通道进行卷积来减少计算量和内存访问，但在特征提取的选择性方面存在不足
。它没有机制能够有效区分哪些特征更加重要，所有的通道均被一视同仁处理，这在面对复杂的视觉任务时可能会影响特征表达能力。
改进方法：
引入通道注意力机制（SE模块）：为了解决特征提取选择性不足的问题，我们引入了通道注意力机制（SE模块），该模块可以动态地
为每个通道分配权重，从而增强重要特征的表达，抑制不重要的特征。
"""
class PartialConv3WithSE(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        # 通道注意力模块（SE模块）
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim, dim // 16, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // 16, dim, bias=False)
        self.sigmoid = nn.Sigmoid()

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()  # 保持原始输入不变
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        # 添加通道注意力
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)  # 全局平均池化并调整形状为 (b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)  # 调整形状为 (b, c, 1, 1)
        x = x * y

        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # 添加通道注意力
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)  # 全局平均池化并调整形状为 (b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)  # 调整形状为 (b, c, 1, 1)
        x = x * y

        return x


if __name__ == '__main__':
    block = PartialConv3WithSE(64, 2, 'split_cat').cuda()
    input_tensor = torch.rand(1, 64, 64, 64).cuda()
    output = block(input_tensor)
    print(input_tensor.size(), output.size())
