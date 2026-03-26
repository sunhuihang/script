import torch
from torch import nn
import torch.nn.functional as F

class DeformableInteractiveAttention(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(DeformableInteractiveAttention, self).__init__()

        # 定义卷积层，将输入通道数从 2 转换为 1
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        # 定义 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
        # 是否启用调制模式
        self.distortionmode = distortionmode
        # 上采样操作，scale_factor=2表示放大两倍
        self.upsample = nn.Upsample(scale_factor=2)
        # 两个下采样卷积层，用于减少特征图尺寸
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        # 如果启用了调制模式
        if distortionmode:
            # 定义调制卷积层，并将其权重初始化为零
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            # 注册后向传播钩子，设置学习率
            self.d_conv.register_full_backward_hook(self._set_lra)

            # 另一个调制卷积层，同样初始化权重为零
            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)

    @staticmethod
    def _set_lra(module, grad_input, grad_output):
        # 设置学习率大小，通过修改梯度来控制更新
        grad_input = [g * 0.4 if g is not None else None for g in grad_input]
        grad_output = [g * 0.4 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input

    @staticmethod
    def _set_lrm(module, grad_input, grad_output):
        # 设置另一种学习率大小，控制不同卷积层的梯度更新
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input

    def forward(self, x):
        # 计算输入张量在第一个维度（通道维）上的均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 对均值和最大值进行下采样
        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)
        
        # 将下采样后的均值和最大值在通道维度拼接
        out = torch.cat([max_out, avg_out], dim=1)

        # 如果启用了调制模式
        if self.distortionmode:
            # 对均值和最大值分别进行卷积，得到调制因子
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))
            d_max_out = torch.sigmoid(self.d_conv1(max_out))
            # 调制最大值和均值
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)

        # 对拼接后的张量进行卷积操作
        out = self.conv(out)
        # 使用上采样操作放大结果，并应用 Sigmoid 激活
        mask = self.sigmoid(self.upsample(out))
        # 通过 mask 对输入张量进行加权
        att_out = x * mask
        # 返回 ReLU 激活后的结果
        return F.relu(att_out)

if __name__ == '__main__':
    # 设置输入张量的尺寸
    B, C, H, W = 1, 32, 256, 256  # 批量大小 B, 输入通道数 C, 高度 H, 宽度 W
    x = torch.randn(B, C, H, W).cuda()  # 创建输入张量，形状为 (B, C, H, W)，并将其移到 GPU

    # 创建 DeformableInteractiveAttention 模型实例
    model = DeformableInteractiveAttention(stride=1, distortionmode=True).cuda()

    # 打印模型结构
    print(model)
    print("哔哩哔哩: CV缝合救星!")

    # 前向传播
    output = model(x)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")  # 打印输入张量的形状
    print(f"输出张量的形状: {output.shape}")  # 打印输出张量的形状
