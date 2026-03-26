import torch
import torch.nn as nn
from einops import rearrange


# 魔改后的TSSA模块，增加了动态调整温度参数的功能
class ModifiedTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., temperature_scale=1.0):
        super().__init__()
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        # 动态温度参数，初始化为可学习参数并乘上缩放因子
        self.temp = nn.Parameter(torch.ones(num_heads, 1) * temperature_scale)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
        # 增加一个卷积层，用于对输入特征进行初步处理
        self.pre_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        if x.dim() == 4:  # 处理图片数据
            x = self.pre_conv(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        b, h, N, d = w.shape
        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2
        # 动态调整Pi的计算，根据输入数据的统计信息调整温度参数
        data_mean = torch.mean(x, dim=(1, 2), keepdim=True)
        new_temp = self.temp * (1 + torch.abs(data_mean))
        Pi = self.attend(torch.sum(w_sq, dim=-1) * new_temp)
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)
        out = -torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if x.dim() == 4:  # 还原图片数据维度
            out = out.view(x.shape[0], -1, x.shape[1], x.shape[2]).transpose(1, 2)
        return self.to_out(out)


if __name__ == "__main__":
    # 创建修改后的TSSA模块实例，64代表通道维度
    modified_TSSA = ModifiedTSSA(64)
    # 1. 处理图片4维数据. CV方向 输入B C H W, 输出B C H W
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(1, 64, 32, 32)
    input1 = input_img
    output = modified_TSSA(input_img)
    output = output.view(1, 64, 32, 32)
    # 输出输入图片张量和输出图片张量的形状
    print("CV_Modified_TSSA_input size:", input1.size())
    print("CV_Modified_TSSA_output size:", output.size())
    print("创新后的TSSA模块在CV任务中运行成功！")
    # 2. 处理3维数据. NLP或时序任务方向 输入B L C, 输出B L C
    B, N, C = 1, 1024, 64  # 批量大小、序列长度、特征维度
    input2 = torch.randn(B, N, C)
    output = modified_TSSA(input2)
    print('NLP_Modified_TSSA_input size:', input2.size())
    print('NLP_Modified_TSSA_output size:', output.size())
    print("创新后的TSSA模块在NLP任务中运行成功！")
