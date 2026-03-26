import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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


class FourierUnit(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        # 注意：这里处理的是实部和虚部拼接后的 2x 通道
        self.conv_layer = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, stride=1,
                                    padding=0, groups=self.groups, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.size()

        # 1. 执行 2D FFT，得到复数形式的频域张量
        ffted = torch.fft.rfft2(x, norm=self.fft_norm)  # [B, C, H, W//2 + 1], complex

        # 2. 拆分实部和虚部，并拼接在 channel 维度上（用于实数卷积处理）
        ffted_real = ffted.real
        ffted_imag = ffted.imag
        ffted_combined = torch.cat([ffted_real, ffted_imag], dim=1)  # [B, 2C, H, W//2+1]

        # 3. 1x1 实数卷积 + 激活
        ffted_out = self.conv_layer(ffted_combined)
        ffted_out = self.act(ffted_out)

        # 4. 分离卷积后的实部和虚部
        out_real, out_imag = torch.chunk(ffted_out, 2, dim=1)
        ffted_complex = torch.complex(out_real, out_imag)  # [B, C, H, W//2+1], complex

        # 5. 逆傅里叶变换：从频域回到空间域
        output = torch.fft.irfft2(ffted_complex, s=(H, W), norm=self.fft_norm)  # [B, C, H, W]
        return output

class FMA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x
        pos_embed = self.CPE(x)
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a_all = torch.split(a, math.ceil(N // 4), dim=-1)
        v_all = torch.split(v, math.ceil(N // 4), dim=-1)
        attns = []
        for a, v in zip(a_all, v_all):
            attn = a * v
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x = x + pos_embed
        x = self.proj(x)
        out = x + shortcut
        
        return out

if __name__ == "__main__":
    batch_size = 1
    channels = 32
    height = 256
    width = 256
    num_heads = 4  # 注意：channels 应当能被 num_heads 整除

    # 输入张量 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)

    # 实例化模型
    model = FMA(dim=channels, num_heads=num_heads)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x)

    # 输出模型结构与形状信息
    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]
