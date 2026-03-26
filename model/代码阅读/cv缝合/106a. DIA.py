import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# 哔哩哔哩CV缝合救星2025.05.27视频

# 权重初始化函数
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif hasattr(m, 'initialize'):
            m.initialize()

# 方向感知门控卷积
class DirectionalGatedConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim)
        self.conv_d = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv_h(x)
        v = self.conv_v(x)
        d = self.conv_d(x)
        g = self.gate(x)
        return g * (h + v + d)

class DirectionalInteractionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q_conv = DirectionalGatedConv(dim)
        self.k_conv = DirectionalGatedConv(dim)
        self.v_conv = DirectionalGatedConv(dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)


if __name__ == "__main__":
    # 输入设置
    batch_size = 1
    channels = 32
    height = 256
    width = 256
    num_heads = 8
    bias = True

    x = torch.randn(batch_size, channels, height, width).cuda()

    model = DirectionalInteractionAttention(dim=channels, num_heads=num_heads, bias=bias).cuda()
    model.initialize()
    print(model)

    output = model(x)

    print("输入形状:", x.shape)
    print("\n哔哩哔哩：CV缝合救星！魔改DIA\n")
    print("输出形状:", output.shape)