import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新2：引入空间注意力
全局信息利用不足：原始模块中，卷积操作仅限于局部空间特征提取，缺乏对全局特征的充分利用。这意味着模型在理解全局上下文时可能存在局限，
尤其在需要捕捉长距离依赖关系的任务中表现欠佳。
改进方法：
1. 在部分卷积后加入空间注意力模块，使得模型能够对特征图的空间位置进行加权，聚焦于对目标更有辨识度的区域。
2. 使用7×7卷积核提取全局空间信息：通过使用大卷积核（7×7）提取空间注意力，可以获取更大的感受野，使模型更加专注于全局空间模式。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv3WithSpatialAttention(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()  # 保持原始输入不变
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        # 添加空间注意力
        attention_map = self.spatial_attention(x)
        x = x * attention_map

        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # 添加空间注意力
        attention_map = self.spatial_attention(x)
        x = x * attention_map

        return x


if __name__ == '__main__':
    block = PartialConv3WithSpatialAttention(64, 2, 'split_cat').cuda()
    input_tensor = torch.rand(1, 64, 64, 64).cuda()
    output = block(input_tensor)
    print(input_tensor.size(), output.size())

