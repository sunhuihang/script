import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 新增位置注意力相关层
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.pos_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.pos_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        # 新增位置注意力计算
        pos_x = x + self.pos_embed
        pos_q = self.pos_q(pos_x)
        pos_k = self.pos_k(pos_x)
        pos_q = rearrange(pos_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        pos_k = rearrange(pos_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        pos_attn = (pos_q @ pos_k.transpose(-2, -1))
        pos_attn = nn.functional.softmax(pos_attn, dim=-1)
        pos_out = (pos_attn @ v)
        pos_out = rearrange(pos_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out + pos_out  # 融合两种注意力结果

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=3, padding=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                bias=bias)
        self.activation = nn.SiLU()  # 更换激活函数
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.activation(x)
        x = self.project_out(x)
        return x


# Lightweight Cross Attention
class LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(LCA, self).__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 新增自适应参数

    def forward(self, x, y):
        norm_x = self.norm(x)
        norm_y = self.norm(y)
        cross_attn_out = self.ffn(norm_x, norm_y)
        x = x + self.alpha * cross_attn_out
        gdfn_out = self.gdfn(self.norm(x))
        x = x + (1 - self.alpha) * gdfn_out
        return x


# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module = LCA(dim=64, num_heads=4)
    input_x = torch.randn(1, 64, 32, 32)
    input_y = torch.randn(1, 64, 32, 32)
    output_tensor = module(input_x, input_y)
    print('LCA_Input size:', input_x.size())
    print('LCA_Output size:', output_tensor.size())
    print('魔改后的LCA模块运行成功！')

    IEL = IEL(dim=64)
    output_tensor = IEL(input_x)
    print('IEL_Input size:', input_x.size())
    print('IEL_Output size:', output_tensor.size())