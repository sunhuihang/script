import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

## Compact Self-Attention (CSA)
# CV缝合救星2025.06.18视频
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
class CompactSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, sample_rate):
        super(CompactSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim//2, dim//2 * 3, kernel_size=1, bias=bias)

        self.sampler = nn.AvgPool2d(1, stride=sample_rate)
        self.kernel_size = sample_rate
        self.patch_size = sample_rate

        self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size=self.kernel_size, padding=(self.kernel_size // sample_rate - 1),
                                            stride=sample_rate, groups=dim, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim//2 * 3, dim//2 * 3, kernel_size=3, stride=1, padding=1, groups=dim//2 * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.sampler(x)

        x1, x2 = x.chunk(2, dim=1)

        b, c, h, w = x1.shape

        ########### produce q1,k1 and v1 from x1 token feature

        qkv_1 = self.qkv_dwconv(self.qkv(x1))
        q1, k1, v1 = qkv_1.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        ########### produce q2,k2 and v2 from x2 token feature

        qkv_2 = self.qkv_dwconv(self.qkv(x2))
        q2, k2, v2 = qkv_2.chunk(3, dim=1)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        ####### cross-token self-attention

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ v2)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)

        out2 = (attn2 @ v1)

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = torch.cat([out1, out2], dim=1)

        out = self.LocalProp(out)

        out = self.project_out(out)

        out = out[:, :, :H, :W]
        return out
    
if __name__ == "__main__":
    # 配置测试参数
    batch_size = 1
    channels = 32
    height = 256
    width = 256
    num_heads = 8     # 注意力头数
    sample_rate = 2    # 下采样率
    bias = True        # 卷积层是否使用偏置

    # 创建随机输入张量 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)
    
    # 实例化注意力模块
    model = CompactSelfAttention(
        dim=channels,
        num_heads=num_heads,
        bias=bias,
        sample_rate=sample_rate
        # CV缝合救星2025.06.18视频
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
    )
    
    # 设备配置 (优先使用GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)
    
    # 前向传播
    output = model(x)
    
    # 打印模型结构和输入输出形状
    print(model)
    print("\n哔哩哔哩: CV缝合救星！\n")
    print("输入形状:", x.shape)      # 应为 [1, 32, 256, 256]
    print("输出形状:", output.shape)  # 应保持与输入相同 [1, 32, 256, 256]
