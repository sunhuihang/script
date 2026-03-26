import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
"""
CV缝合救星魔改创新1：引入可学习的边界权重机制 (Learnable Boundary Weight Mechanism)
背景：目前的 SBA 模块通过 Sigmoid 激活来生成静态的边界权重，这虽然能够一定程度上调节低层和高层特征的融合，
但并不能灵活地学习每个区域的边界重要性。
创新点：引入一个可学习的边界权重机制，使得模型能够动态地学习在每个空间位置上的边界重要性。这是通过一个小的
卷积层来实现的，生成一个权重图，控制不同空间区域的特征融合程度。
具体实现：
1.可学习边界权重图：通过一个简单的卷积层（如3x3卷积）来学习每个空间位置的边界权重。这些权重将与特征图进行逐元素乘法，
强化模型对边界区域的关注。
2. 动态边界调节：模型可以根据学习到的权重来动态地调整低层和高层特征的融合程度，而不是依赖于静态的Sigmoid 权重。
"""
warnings.filterwarnings('ignore')

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 简化的SBA模块，并添加可学习的边界权重机制
class SBA(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(SBA, self).__init__()

        # 将输入通道数减半，并通过1x1卷积进行通道数的调整
        self.fc1 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)

        # 通过简单的卷积层增强特征
        self.d_in1 = BasicConv2d(input_dim // 2, input_dim // 2, kernel_size=1)
        self.d_in2 = BasicConv2d(input_dim // 2, input_dim // 2, kernel_size=1)

        # 输出卷积
        self.conv = nn.Sequential(
            BasicConv2d(input_dim, input_dim, 3, 1, 1),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        )

        # 生成边界权重图的可学习卷积层，输入通道修改为 input_dim//2
        self.boundary_weight_conv = nn.Conv2d(input_dim // 2, 1, kernel_size=3, padding=1, bias=False)

        # Sigmoid激活函数
        self.Sigmoid = nn.Sigmoid()

    def forward(self, H_feature, L_feature):
        # 将低层和高层特征进行通道调整
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        # 使用Sigmoid生成基础权重
        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        # 通过卷积层生成可学习的边界权重图
        boundary_weight = self.boundary_weight_conv(L_feature)  # Shape: (B, 1, H, W)
        boundary_weight = self.Sigmoid(boundary_weight)  # 生成边界权重的sigmoid输出

        # 进行1x1卷积操作
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        # 进行特征融合，同时结合边界权重
        L_feature = L_feature + boundary_weight * (L_feature * g_L_feature)
        H_feature = H_feature + boundary_weight * (H_feature * g_H_feature)

        # 融合后的高低层特征
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

# 测试代码
if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64) # x: (B, C,H, W)
    input2 = torch.randn(1, 32, 64, 64) # x: (B, C,H, W)
    model = SBA(input_dim=32, output_dim=32)
    output = model(input1, input2)
    print("SBA_input size:", input1.size())
    print("SBA_Output size:", output.size())
