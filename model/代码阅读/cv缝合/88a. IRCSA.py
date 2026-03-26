import torch
import torch.nn as nn
from torch.nn import Softmax
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class InterRowColSelfAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini, axis='H'):
        """
        初始化方法，定义了卷积层和位置嵌入。
        Parameters:
        in_dim : int  # 输入张量的通道数
        q_k_dim : int  # Q 和 K 向量的通道数
        axis : str  # 注意力计算的轴 ('H', 'W')
        """
        super(InterRowColSelfAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis
        H, W = patch_ini[0], patch_ini[1]

        # 定义卷积层
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # 根据轴选择不同的位置信息嵌入
        if self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, H, 1))  # 高度方向嵌入
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, W))  # 宽度方向嵌入
        else:
            raise ValueError("Axis must be one of 'H' or 'W'.")  # 如果轴不是 'H', 'W' 则报错

        # 使用 Xavier 初始化位置嵌入
        nn.init.xavier_uniform_(self.pos_embed)

        self.softmax = Softmax(dim=-1)  # 定义 softmax 层
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义可训练的缩放参数
        self.ca = ChannelAttention(in_dim)

    def forward(self, x, processed):
        """
        前向传播方法，计算注意力机制。
        参数：
        x : Tensor  # 输入的 4D 张量 (batch, channels, height, width)
        processed : Tensor  # 处理过的输入张量，形状与 x 相同
        """
        B, C, H, W = x.size()

        # 计算 Q, K, V
        Q = self.query_conv(processed) + self.pos_embed  # (B, q_k_dim, H, W) + pos_embed
        K = self.key_conv(processed) + self.pos_embed  # (B, q_k_dim, H, W) + pos_embed
        V = self.value_conv(processed)  # (B, in_dim, H, W)
        scale = math.sqrt(self.q_k_dim)  # 缩放因子

        # 根据注意力轴 ('H', 'W') 进行不同维度的处理
        if self.axis == 'H':  # 如果是高度方向
            Q = Q.permute(0, 2, 3, 1).contiguous()  # 重新排列维度为 (B, H, W, q_k_dim)
            Q = Q.view(B * W, H, self.q_k_dim)  # 展平为 (B*W, H, q_k_dim)

            K = K.permute(0, 2, 3, 1).contiguous()
            K = K.view(B * W, H, self.q_k_dim).permute(0, 2, 1).contiguous()  # 展平为 (B*W, q_k_dim, H)

            V = V.permute(0, 2, 3, 1).contiguous()
            V = V.view(B * W, H, self.in_dim)  # 展平为 (B*W, H, in_dim)

            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*W, H, H)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*W, H, in_dim)
            out = out.view(B, W, H, self.in_dim).permute(0, 3, 2, 1).contiguous()  # 最终输出形状 (B, C, H, W)

        else:  # 如果是宽度方向
            Q = Q.permute(0, 2, 3, 1).contiguous()  # 重新排列维度为 (B, H, W, q_k_dim)
            Q = Q.view(B * H, W, self.q_k_dim)  # 展平为 (B*H, W, q_k_dim)

            K = K.permute(0, 2, 3, 1).contiguous()
            K = K.view(B * H, W, self.q_k_dim).permute(0, 2, 1).contiguous()  # 展平为 (B*H, q_k_dim, W)

            V = V.permute(0, 2, 3, 1).contiguous()
            V = V.view(B * H, W, self.in_dim)  # 展平为 (B*H, W, in_dim)

            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*H, W, W)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*H, W, in_dim)
            out = out.view(B, H, W, self.in_dim).permute(0, 3, 1, 2).contiguous()  # 最终输出形状 (B, C, H, W)

        # 使用 gamma 融合输入和输出
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1 - gamma) * x  # 输出加权

        # 加入通道注意力
        ca_out = self.ca(out)
        out = out * ca_out

        return out


if __name__ == '__main__':
    # 设置输入参数
    batch_size = 1  # 批次大小
    in_channels = 32  # 输入通道数
    q_k_dim = 16  # Q, K 向量的通道数
    input_resolution = (64, 64)  # 输入张量的分辨率
    axis = 'H'  # 在高度方向进行注意力操作

    # 创建随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1]).cuda()
    processed = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1]).cuda()

    # 创建 InterRowColSelfAttention 模块
    model = InterRowColSelfAttention(in_dim=in_channels, q_k_dim=q_k_dim, patch_ini=input_resolution, axis=axis).cuda()

    # 打印模型结构
    print(model)

    # 前向传播
    output = model(x, processed)

    # 打印输入和输出张量的形状
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
