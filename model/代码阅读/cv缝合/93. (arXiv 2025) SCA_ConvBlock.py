import torch
import torch.nn as nn

class SCA_ConvBlock(nn.Module):
    """
    Simplified Channel Attention (SCA) ConvBlock: A simplified channel attention
    block for faster computation.

    c : int, Number of input and output channels for the block.
    """

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.fq = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fk = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        fq = self.fq(inputs)
        fk = self.fk(inputs)
        fv = self.fv(inputs)

        f_sim_tensor = torch.matmul(fq, fk.transpose(2, 3)) / (fq.size(-1) ** 0.5)
        f_sum_tensor = torch.sum(f_sim_tensor, dim=(2, 3))
        scores = torch.softmax(f_sum_tensor, dim=1).unsqueeze(2).unsqueeze(3)

        r = scores * fv + inputs
        r = self.bn(r)
        r = self.relu(r)
        return r
    
if __name__ == "__main__":
    batch_size = 1  # Batch size
    channels = 32   # 输入通道数
    height = 256    # 输入图像高度
    width = 256     # 输入图像宽度

    # 创建一个模拟输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化 SCA_ConvBlock 模块
    model = SCA_ConvBlock(c = channels)
    print(model)
    print("哔哩哔哩：CV缝合救星!")
    # 前向传播
    output = model(x)

    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")