import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class FrequencyGuidedChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fft_feat = torch.fft.fft2(x, norm='ortho')
        fft_mag = torch.abs(torch.fft.fftshift(fft_feat, dim=(-2, -1)))
        descriptor = self.pool(fft_mag).view(x.size(0), -1)
        weight = self.fc(descriptor).view(x.size(0), x.size(1), 1, 1)
        return x * weight


class FrequencyDynamicConv(nn.Module):
    def __init__(self, in_channels, kernel_choices=3):
        super().__init__()
        self.kernel_choices = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
            for _ in range(kernel_choices)
        ])
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, kernel_choices, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        attn_logits = self.attn(x)                      # (B, K, 1, 1)
        attn_weights = F.softmax(attn_logits, dim=1)    # softmax over K
        out = 0
        for i, conv in enumerate(self.kernel_choices):
            out += attn_weights[:, i:i+1] * conv(x)
        return out


class FGCBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fcpe = nn.Conv2d(2 * in_channels, 2 * in_channels, 3, padding=1, groups=2 * in_channels)
        self.fdc = FrequencyDynamicConv(2 * in_channels)
        self.fgca = FrequencyGuidedChannelAttention(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        fft_x = torch.fft.fft2(x, norm='ortho')
        FR = fft_x.real
        FI = fft_x.imag
        FJ = torch.cat([FR, FI], dim=1)                 # (B, 2C, H, W)

        FJ = FJ + self.fcpe(FJ)
        FJ = self.fdc(FJ)

        FR, FI = torch.chunk(FJ, 2, dim=1)
        fft_out = torch.complex(FR, FI)
        spatial_out = torch.fft.ifft2(fft_out, norm='ortho').real  # (B, C, H, W)

        out = self.fgca(spatial_out)
        return out


# ============ ✅ 主函数 ============
if __name__ == "__main__":
    input_tensor = torch.randn(2, 64, 64, 64)  # (B=2, C=64, H=64, W=64)
    block = FGCBlock(in_channels=64)
    output = block(input_tensor)
    print(block)
    print("Input shape:", input_tensor.shape)
    print("\n哔哩哔哩：CV缝合救星!\n")
    print("Output shape:", output.shape)
