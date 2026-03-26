import torch
import torch.nn as nn
from einops import rearrange
# 哔哩哔哩：CV缝合救星
class TSSA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)# 哔哩哔哩：CV缝合救星

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        b, h, N, d = w.shape

        w_normed = torch.nn.functional.normalize(w, dim=-2)# 哔哩哔哩：CV缝合救星
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)# 哔哩哔哩：CV缝合救星
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


if __name__ == "__main__":
    #创建TSSA模块实例，64代表通道维度
    TSSA = TSSA(64)

    # 1.如何输入的是图片4维数据. CV方向  输入 B C H W, 输出 B C H W
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(1, 64, 32, 32)
    input1 = input_img
    input_img = input_img.reshape(1, 64, -1).transpose(-1, -2)
    # 运行前向传递
    output = TSSA(input_img)
    output = output.view(1, 64, 32, 32)  # 将三维度转化成图片四维度张量
    # 输出输入图片张量和输出图片张量的形状
    print("CV_TSSA_input size:", input1.size())
    print("CV_TSSA_output size:", output.size())

    print("哔哩哔哩：CV缝合救星！")

    # 2.如何输入的3维数据 . NLP或时序任务方向  输入 B L C, 输出 B L C
    B, N, C = 1, 1024, 64  # 批量大小、序列长度、特征维度
    input2 = torch.randn(B, N, C)
    output = TSSA(input2)
    print('NLP_TSSA_input size:',input2.size())
    print('NLP_TSSA_output size:',output.size())