import torch
import torch.nn as nn
from torch.nn import Softmax
import math

class InterSliceSelfAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini, axis='D'):
        """
        初始化方法，定义了卷积层和位置嵌入。
        Parameters:
        in_dim : int  # 输入张量的通道数
        q_k_dim : int  # Q 和 K 向量的通道数
        axis : str  # 注意力计算的轴 ('D', 'H', 'W')
        """
        super(InterSliceSelfAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis
        D, H, W = patch_ini[0], patch_ini[1], patch_ini[2]

        # 定义卷积层
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # 根据轴选择不同的位置信息嵌入
        if self.axis == 'D':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, D, 1, 1))  # 深度方向嵌入
        elif self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, H, 1))  # 高度方向嵌入
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, 1, W))  # 宽度方向嵌入
        else:
            raise ValueError("Axis must be one of 'D', 'H', or 'W'.")  # 如果轴不是 'D', 'H', 'W' 则报错

        # 使用 Xavier 初始化位置嵌入
        nn.init.xavier_uniform_(self.pos_embed)
        
        self.softmax = Softmax(dim=-1)  # 定义 softmax 层
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义可训练的缩放参数

    def forward(self, x, processed):
        """
        前向传播方法，计算注意力机制。
        参数：
        x : Tensor  # 输入的 5D 张量 (batch, channels, depth, height, width)
        processed : Tensor  # 处理过的输入张量，形状与 x 相同
        """
        B, C, D, H, W = x.size()

        # 计算 Q, K, V
        Q = self.query_conv(processed) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        K = self.key_conv(processed) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        V = self.value_conv(processed)  # (B, in_dim, D, H, W)
        scale = math.sqrt(self.q_k_dim)  # 缩放因子

        # 根据注意力轴 ('D', 'H', 'W') 进行不同维度的处理
        if self.axis == 'D':  # 如果是深度方向
            Q = Q.permute(0, 3, 4, 2, 1).contiguous()  # 重新排列维度为 (B, H, W, D, q_k_dim)
            Q = Q.view(B*H*W, D, self.q_k_dim)  # 展平为 (B*H*W, D, q_k_dim)
            
            K = K.permute(0, 3, 4, 1, 2).contiguous()  # 重新排列维度为 (B, H, W, q_k_dim, D)
            K = K.view(B*H*W, self.q_k_dim, D)  # 展平为 (B*H*W, q_k_dim, D)
            
            V = V.permute(0, 3, 4, 2, 1).contiguous()  # 重新排列维度为 (B, H, W, D, in_dim)
            V = V.view(B*H*W, D, self.in_dim)  # 展平为 (B*H*W, D, in_dim)
            
            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*H*W, D, D)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*H*W, D, in_dim)
            out = out.view(B, H, W, D, self.in_dim)  # 恢复为原始形状 (B, H, W, D, in_dim)
            out = out.permute(0, 4, 3, 1, 2).contiguous()  # 最终输出形状 (B, C, D, H, W)
        
        elif self.axis == 'H':  # 如果是高度方向
            Q = Q.permute(0, 2, 4, 3, 1).contiguous()  # 重新排列维度为 (B, D, W, H, q_k_dim)
            Q = Q.view(B*D*W, H, self.q_k_dim)  # 展平为 (B*D*W, H, q_k_dim)
            
            K = K.permute(0, 2, 4, 1, 3).contiguous()  # 重新排列维度为 (B, D, W, q_k_dim, H)
            K = K.view(B*D*W, self.q_k_dim, H)  # 展平为 (B*D*W, q_k_dim, H)
            
            V = V.permute(0, 2, 4, 3, 1).contiguous()  # 重新排列维度为 (B, D, W, H, in_dim)
            V = V.view(B*D*W, H, self.in_dim)  # 展平为 (B*D*W, H, in_dim)
            
            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*D*W, H, H)
            attn = self.softmax(attn)  # 进行 softmax 操作
            
            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*D*W, H, in_dim)
            out = out.view(B, D, W, H, self.in_dim)  # 恢复为原始形状 (B, D, W, H, in_dim)
            out = out.permute(0, 4, 1, 3, 2).contiguous()  # 最终输出形状 (B, C, D, H, W)
        
        else:  # 如果是宽度方向
            Q = Q.permute(0, 2, 3, 4, 1).contiguous()  # 重新排列维度为 (B, D, H, W, q_k_dim)
            Q = Q.view(B*D*H, W, self.q_k_dim)  # 展平为 (B*D*H, W, q_k_dim)
            
            K = K.permute(0, 2, 3, 1, 4).contiguous()  # 重新排列维度为 (B, D, H, q_k_dim, W)
            K = K.view(B*D*H, self.q_k_dim, W)  # 展平为 (B*D*H, q_k_dim, W)
            
            V = V.permute(0, 2, 3, 4, 1).contiguous()  # 重新排列维度为 (B, D, H, W, in_dim)
            V = V.view(B*D*H, W, self.in_dim)  # 展平为 (B*D*H, W, in_dim)
            
            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*D*H, W, W)
            attn = self.softmax(attn)  # 进行 softmax 操作
            
            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*D*H, W, in_dim)
            out = out.view(B, D, H, W, self.in_dim)  # 恢复为原始形状 (B, D, H, W, in_dim)
            out = out.permute(0, 4, 1, 2, 3).contiguous()  # 最终输出形状 (B, C, D, H, W)

        # 使用 gamma 融合输入和输出
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1 - gamma) * x  # 输出加权
        return out

if __name__ == '__main__':
    # 设置输入参数
    batch_size = 1          # 批次大小
    in_channels = 32        # 输入通道数
    q_k_dim = 16            # Q, K 向量的通道数
    input_resolution = (64, 64, 64)  # 输入张量的分辨率
    axis = 'D'  # 在深度方向进行注意力操作
    
    # 创建随机输入张量 (batch_size, channels, depth, height, width)
    x = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1], input_resolution[2]).cuda()
    processed = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1], input_resolution[2]).cuda()
    
    # 创建 InterSliceSelfAttention 模块
    model = InterSliceSelfAttention(in_dim=in_channels, q_k_dim=q_k_dim, patch_ini=input_resolution, axis=axis).cuda()

    # 打印模型结构
    print(model)
    print("哔哩哔哩: CV缝合救星!")
    # 前向传播
    output = model(x, processed)

    # 打印输入和输出张量的形状
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")