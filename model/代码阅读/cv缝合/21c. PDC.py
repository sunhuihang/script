import torch
from torch import nn
import torch.nn.functional as F


class PartialConv3Plus(nn.Module):
    def __init__(self, dim, n_div=2, forward_mode='split_cat', use_se=True):
        super().__init__()
        self.dim_conv3 = dim // n_div  # 选取部分通道进行卷积
        self.dim_untouched = dim - self.dim_conv3  # 保持不变的通道

        # 使用 Depthwise Separable Convolution 代替所有普通卷积
        self.partial_conv3 = nn.Sequential(
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, groups=self.dim_conv3, bias=False),
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 1, 1, 0, bias=False)
        )

        # SE注意力模块（可选），自适应调整通道权重
        self.use_se = use_se
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 16, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 16, dim, 1, bias=False),
                nn.Sigmoid()
            )

        # 多尺度特征融合（3x3, 5x5, 7x7）均使用深度可分离卷积
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 5, 1, 2, groups=self.dim_conv3, bias=False),
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 1, 1, 0, bias=False)
        )

        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 7, 1, 3, groups=self.dim_conv3, bias=False),
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 1, 1, 0, bias=False)
        )

        if forward_mode == 'slicing':
            self.forward_mode = self.forward_slicing
        elif forward_mode == 'split_cat':
            self.forward_mode = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1_3x3 = self.partial_conv3(x1)
        x1_5x5 = self.conv_5x5(x1)
        x1_7x7 = self.conv_7x7(x1)
        x1 = x1_3x3 + x1_5x5 + x1_7x7  # 多尺度融合
        x = torch.cat((x1, x2), 1)

        if self.use_se:
            se_weight = self.se(x)
            x = x * se_weight  # SE注意力调节

        return x

    def forward(self, x):
        return self.forward_mode(x)


if __name__ == '__main__':
    block = PartialConv3Plus(64, n_div=2, forward_mode='split_cat').cuda()
    input_tensor = torch.rand(1, 64, 64, 64).cuda()
    output = block(input_tensor)
    print(input_tensor.size(), output.size())
