import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义SPR-SA模块
class SPRSA(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        """
        初始化SPR-SA模块
        
        dim: 输入特征的通道数
        growth_rate: 隐藏层通道数的增长倍率，默认为2.0
        """
        super(SPRSA, self).__init__()

        # 计算隐藏层通道数
        hidden_dim = int(dim * growth_rate)

        # 第一个卷积层：深度可分卷积（Depthwise Convolution），用于局部特征提取
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),  # 深度可分卷积，groups=dim表示每个通道独立卷积
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)  # 1x1卷积，用于通道间信息的融合
        )
        
        # 激活函数，使用GELU（Gaussian Error Linear Unit）
        self.act = nn.GELU()

        # 第二个卷积层，用于输出恢复到原始通道数
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        """
        前向传播

        x: 输入特征图，形状为 (batch_size, channels, height, width)
        返回：输出特征图，形状为 (batch_size, channels, height, width)
        """
        # 通过conv_0提取局部特征
        x = self.conv_0(x)
        
        # CV缝合救星：此时图像经过卷积后，需要进行空间上的调整，利用全局池化（global pooling）来处理局部信息
        x1 = F.adaptive_avg_pool2d(x, (1, 1))  # 自适应平均池化，将特征图池化为1x1
        x1 = F.softmax(x1, dim=1)  # 对通道维度进行softmax操作，类似于特征的注意力机制
        
        # 关键步骤：通过软加权对原始特征图进行调整，聚焦重要区域
        x = x1 * x  # 将权重应用到输入特征图
        
        # 激活函数：增加非线性
        x = self.act(x)
        
        # 通过conv_1恢复原始的通道数
        x = self.conv_1(x)
        
        return x

# 测试模型
if __name__ == "__main__":
    # 模拟输入数据，假设batch_size为1，通道数为64，图像大小为32x32
    x = torch.randn(1, 64, 32, 32)  # 随机生成一个输入特征图

    # 创建SPR-SA模块，dim设为64（输入通道数）
    model = SPRSA(dim=64, growth_rate=2.0)

    # 打印模型结构
    print("SPR-SA模型结构：")
    print(model)

    # 测试前向传播
    output = model(x)

    # 打印输入输出形状
    print("\n输入形状：", x.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星独家复现\n")
    print("输出形状：", output.shape)
