import torch
import torch.nn as nn

class PTeLU(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, learnable=False):
        super(PTeLU, self).__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
            self.register_buffer('beta', torch.tensor(beta))

    def forward(self, x):
        return self.beta * x * torch.tanh(torch.exp(self.alpha * x))

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256).cuda()

    # 可设置为 learnable=True 开启训练自适应
    telu = PTeLU(alpha=1.5, beta=0.8, learnable=False).cuda()

    output = telu(x)

    print("\nP-TeLU: \n哔哩哔哩：CV缝合救星：让激活函数“动”起来！\n")
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
