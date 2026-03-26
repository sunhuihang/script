import torch
import torch.nn as nn
import torch.fft as fft
"""
创新点 6：使用空间频域处理 (Spatial Frequency Domain Processing)
一、背景
现有的空间注意力图通常依赖于时域特征（如平均池化、最大池化等），这些方法着重于捕捉图像的低层次信息。
然而，在某些任务中，高频信息（如纹理、边缘等）对于区分细节和语义至关重要。因此，时域处理方法可能无
法有效地捕捉到这些高频特征。

二、实现方法
1. 引入频域处理：
通过傅里叶变换等频域操作，将图像从时域转换到频域，提取出图像的频率信息。频域中的高频成分通常包含图像
的细节信息，如纹理、边缘等，这些信息在空间域中可能较难捕捉到。
2. 频域特征与时域特征结合：
在空间注意力图生成过程中，将频域提取的高频信息与传统的时域特征（如池化特征）进行结合，生成更为全面的
空间注意力图。这样做可以帮助模型更好地捕捉图像中的细节特征，尤其是在纹理、边缘等重要细节上。
3. 频域注意力调整：
使用频域信息生成的注意力图对原始特征图进行加权调整，不仅关注图像的低层次信息，还加强对高频信息的学习。
通过这种方式，模型能够同时捕捉全局和细节信息，提升模型的感知能力。
"""
class SpatialAttentionWithFrequency(nn.Module):
    def __init__(self, in_channels):
        """
        使用频域处理增强空间注意力图
        :param in_channels: 输入的通道数
        """
        super(SpatialAttentionWithFrequency, self).__init__()

        # 空间注意力卷积层
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3)

        # Sigmoid激活生成空间注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播，生成通过频域处理增强的空间注意力图
        :param x: 输入特征图
        :return: 生成的空间注意力图
        """
        batch_size, channels, height, width = x.size()

        # 时域空间注意力图生成（通过卷积）
        spatial_attention = self.conv1(x)

        # 使用傅里叶变换提取频域信息
        x_freq = fft.fft2(x)  # 进行二维傅里叶变换
        x_freq = torch.abs(x_freq)  # 获取频域的幅度谱（幅度可以反映细节信息）

        # 将频域信息与时域注意力图相结合
        spatial_attention = spatial_attention + x_freq.mean(dim=1, keepdim=True)  # 平均频域信息，保持通道维度

        # 使用Sigmoid激活生成空间注意力图
        attention_map = self.sigmoid(spatial_attention)

        # 按照注意力图加权输入
        output = x * attention_map
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
    spatial_attention_freq = SpatialAttentionWithFrequency(in_channels=64)
    output_tensor = spatial_attention_freq(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
