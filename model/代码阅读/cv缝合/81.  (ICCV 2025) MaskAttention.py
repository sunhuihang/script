import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskAttention(nn.Module):
    def __init__(self, channels, size):
        super(MaskAttention, self).__init__()
        self.channels = channels  # 输入的通道数
        self.size = size  # 输入图像的大小（height, width）
        
        # 定义查询（Q）、键（K）和值（V）的线性变换层
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        self.mask = None  # 初始化mask，暂时为空
        self.norm = nn.LayerNorm([channels])  # 对输出进行LayerNorm标准化

    def forward(self, x):
        # 获取输入张量的批次大小、通道数、高度和宽度
        batch_size, channels, height, width = x.size()
        
        # 检查输入的通道数是否与初始化时的通道数一致
        if channels != self.channels:
            raise ValueError("Input channel size does not match initialized channel size.")
        
        # 将输入张量展平并转置为 (batch_size, height * width, channels) 形状
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        # 计算查询（Q）、键（K）和值（V）
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数（scaled dot-product）
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.channels ** 0.5)  # 对分数进行缩放

        # 如果mask为空或其尺寸与输入图像大小不匹配，则生成新的随机二值mask
        if self.mask is None or self.mask.size(-1) != height * width:
            # 创建一个二值mask，随机生成0或1的值
            binary_mask = torch.randint(0, 2, (batch_size, height, width), device=x.device)
            binary_mask = binary_mask.view(batch_size, -1)
            
            # 将大于0.5的值设为0，小于0.5的值设为负无穷（即遮蔽这些区域）
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0, device=x.device), torch.tensor(-float('inf'), device=x.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1, height * width, -1)  # 扩展为 (batch_size, height * width, height * width) 形状
        
        # 将mask添加到注意力分数中
        scores = scores + self.mask

        # 对分数进行softmax归一化，得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 使用注意力权重对值（V）进行加权求和，得到注意力输出
        attention_output = torch.matmul(attention_weights, V)

        # 将输入与注意力输出相加，并进行标准化
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)

        # 恢复为原始的 (batch_size, channels, height, width) 形状
        return attention_output.view(batch_size, channels, height, width)
    
if __name__ == "__main__":
    # 设置输入参数
    batch_size = 1  # 批次大小
    channels = 64   # 输入通道数
    height = 16     # 图像的高度
    width = 16      # 图像的宽度

    # 创建随机输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width).cuda()  # 将输入张量移动到 GPU 上

    # 创建 MaskAttention 模块实例
    attention_module = MaskAttention(channels=channels, size=(height, width)).cuda()

    # 打印模型结构
    print("MaskAttention模型结构:")
    print(attention_module)

    # 前向传播
    output = attention_module(x)

    # 打印输入和输出张量的形状
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")
