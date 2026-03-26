import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
CV缝合救星魔改创新：Dynamic Scaling Differential Attention (DSDA)
引入可学习的注意力缩放机制
1. 问题：
当前的注意力机制使用固定的缩放因子（如头部维度的平方根）对查询（Q）和键（K）的点积进行归一化。
这种方法缺乏灵活性，不能适应不同任务的特定特征分布，可能导致注意力分布的过度集中或稀疏。
2. 改进思路：
在 MultiHeadDifferentialAttention 模块中引入可学习的缩放机制，通过参数 alpha 和 beta 动
态调整注意力分数的幅度和偏移。
a. 为每个注意力头分配独立的缩放参数 alpha（控制幅度）和 beta（控制偏移）。
b. 使用梯度学习动态调整这些参数，以适应任务的特定需求。
c. 调整注意力得分：A=A*alpha+beta,该公式能够显式控制注意力分布的强弱，提升模型的灵活性和鲁棒
性。
保持结构一致性：
"""
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        g = F.silu(self.WG(x))
        z = self.W1(x)
        return self.W2(g * z)

class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # New learnable parameters for scaling attention scores
        self.alpha = nn.Parameter(torch.ones(num_heads))
        self.beta = nn.Parameter(torch.zeros(num_heads))

        self.eps = 1e-5

    def forward(self, X):
        batch, N, d_model = X.shape

        Q = self.W_q(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = self.W_k(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = self.W_v(X).view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)

        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling

        # Learnable scaling of attention scores
        A1 = A1 * self.alpha.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        A2 = A2 * self.alpha.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)

        mask = torch.tril(torch.ones((N, N), device=X.device)).unsqueeze(0).unsqueeze(0)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

        A1 = A1 + mask
        A2 = A2 + mask

        attention1 = F.softmax(A1, dim=-1)
        attention2 = F.softmax(A2, dim=-1)
        attention = attention1 - attention2

        O = torch.matmul(attention, V)
        O = O.transpose(1, 2).contiguous().view(batch, N, -1)

        return self.W_o(O)

class DiffTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x):
        y = self.attn(self.norm1(x)) + x
        z = self.ff(self.norm2(y)) + y
        return z

if __name__ == '__main__':
    d_model = 512
    num_heads = 8

    # CV Example
    input_img = torch.randn(1, 512, 32, 32)
    input_cv = input_img.reshape(1, 512, -1).transpose(-1, -2)
    layer = DiffTransformerLayer(d_model, num_heads)
    output_cv = layer(input_cv)
    output_img = output_cv.view(1, 512, 32, 32)
    print("CV Input size:", input_img.size())
    print("CV Output size:", output_img.size())

    # NLP Example
    input_nlp = torch.randn(1, 1024, 512)
    output_nlp = layer(input_nlp)
    print("NLP Input size:", input_nlp.size())
    print("NLP Output size:", output_nlp.size())
