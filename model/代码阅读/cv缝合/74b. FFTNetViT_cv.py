import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))


class FFTNetBlock(nn.Module):
    def __init__(self, channels, window_size=16):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.global_filter = nn.Linear(channels, channels)
        self.local_filter = nn.Linear(channels, channels)
        self.gate = nn.Linear(2 * channels, 1)
        self.modrelu = ModReLU(channels)

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        B, C, H, W = x.shape
        seq_len = H * W

        # 调整为序列形式
        x_seq = x.view(B, C, seq_len).permute(0, 2, 1)

        # 全局傅里叶变换
        x_fft_global = torch.fft.fft(x_seq, dim=1)
        x_filtered_global = self.global_filter(x_fft_global.real) + 1j * self.global_filter(x_fft_global.imag)

        # 局部窗口化（STFT）
        num_windows = seq_len // self.window_size
        x_local = x_seq.view(B, num_windows, self.window_size, self.channels)
        x_fft_local = torch.fft.fft(x_local, dim=2)
        x_fft_local = x_fft_local.view(B, seq_len, self.channels)
        x_filtered_local = self.local_filter(x_fft_local.real) + 1j * self.local_filter(x_fft_local.imag)

        # 门控融合
        gate_input = torch.cat([x_filtered_global.real, x_filtered_local.real], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        x_filtered = gate * x_filtered_global + (1 - gate) * x_filtered_local

        x_filtered = self.modrelu(x_filtered)
        x_out = torch.fft.ifft(x_filtered, dim=1).real

        # 调整回图像形式
        x_out = x_out.permute(0, 2, 1).view(B, C, H, W)
        return x_out


if __name__ == '__main__':
    # 参数设置
    batch_size = 1  # 批量大小
    channels = 32  # 通道数
    height = 224
    width = 224
    window_size = 16  # 局部窗口大小

    # 创建随机输入张量,形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化 FFTNetBlock 模块
    model = FFTNetBlock(channels=channels, window_size=window_size)
    print(model)
    print("哔哩哔哩: CV缝合救星!")
    output = model(x)
    print(x.shape)
    print(output.shape)
