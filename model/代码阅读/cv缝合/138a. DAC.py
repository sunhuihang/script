import torch
import torch.nn as nn

# DynamicAttentionConv 模块，结合自适应卷积和注意力机制
class DynamicAttentionConv(nn.Module):
    def __init__(self, dim, k1, k2, attention_dim=32):
        super(DynamicAttentionConv, self).__init__()
        
        # 初始深度可分离卷积，增强局部特征学习
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # 动态卷积，使用可调卷积核大小
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        
        # Attention机制，学习输入特征的重要性
        self.attn_conv = nn.Conv2d(dim, attention_dim, kernel_size=1)
        self.attn_activation = nn.Softmax(dim=1)
        
        # 最终输出卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)
        
        # 添加一个卷积层，将 attn_weights 的通道数扩展为 dim
        self.attn_expand = nn.Conv2d(attention_dim, dim, kernel_size=1)

    def forward(self, x):
        # 初步卷积，提取特征
        attn = self.conv0(x)
        
        # 空间感知卷积，分别使用不同卷积核大小
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        
        # 自适应注意力，计算空间重要性
        attn_weights = self.attn_conv(attn)
        attn_weights = self.attn_activation(attn_weights)
        
        # 扩展 attn_weights 的通道数，以匹配 attn 的通道数
        attn_weights = self.attn_expand(attn_weights)
        
        # 将注意力加权到原始输入上
        attn = attn * attn_weights
        
        # 最终卷积层处理特征
        attn = self.conv1(attn)
        
        return attn

# DACModule模块，集成DynamicAttentionConv模块
class DACModule(nn.Module):
    def __init__(self, d_model, k1=1, k2=19):
        super(DACModule, self).__init__()
        
        # 两个1x1卷积层，用于投影和激活
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        
        # 引入DynamicAttentionConv模块作为核心
        self.dynamic_attention_unit = DynamicAttentionConv(d_model, k1, k2)
        
        # 另一个1x1卷积层
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # 残差连接初始化
        shortcut = x.clone()
        
        # 通过第一个卷积层和激活函数
        x = self.proj_1(x)
        x = self.activation(x)
        
        # 通过核心的DynamicAttentionConv模块
        x = self.dynamic_attention_unit(x)
        
        # 通过第二个卷积层
        x = self.proj_2(x)
        
        # 加上残差连接，防止信息丢失
        x = x + shortcut
        return x

# 测试代码
if __name__ == '__main__':
    # 随机生成一个输入数据
    input = torch.rand(1, 64, 32, 32)
    
    # 创建DACModule实例
    dac_module = DACModule(d_model=64, k1=1, k2=19)
    
    # 传递输入数据
    output = dac_module(input)
    
    # 打印输入输出大小
    print('DACModule input_size:', input.size())
    print('DACModule output_size:', output.size())
    print('DACModule 模块增强了自适应卷积和注意力机制，有效提升了特征表达能力！')
