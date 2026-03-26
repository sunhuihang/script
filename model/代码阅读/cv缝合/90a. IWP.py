import torch
import torch.nn as nn
import pywt
import numpy as np
import math
from torch.autograd import Function


class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()),
                           torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()),
                           torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L),
                               torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self, wavename):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        self.band_length = len(self.band_low)
        self.band_length_half = self.band_length // 2

    def get_matrix(self):
        L1 = max(self.input_height, self.input_width)
        L = math.floor(L1 / 2)
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        if self.input_height % 2 == 0:
            matrix_h = np.zeros((L, L1 + self.band_length - 2))
        else:
            matrix_h = np.zeros((L + 1, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[:math.floor(self.input_height / 2 + 1), :self.input_height + self.band_length - 2]
        matrix_h_1 = matrix_h[:math.floor(self.input_width / 2 + 1), :self.input_width + self.band_length - 2]
        index = 0
        for i in range(L1 - L - 1):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[:math.floor(self.input_height / 2 + 1), :self.input_height + self.band_length - 2]
        matrix_g_1 = matrix_g[:math.floor(self.input_width / 2 + 1), :self.input_width + self.band_length - 2]
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1[:, (self.band_length_half - 1):end])
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1[:, (self.band_length_half - 1):end])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=torch.float32, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=torch.float32, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=torch.float32, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=torch.float32, device=device)

    def forward(self, input):
        assert len(input.size()) == 4
        self.input = input
        self.input_height = input.size(-2)
        self.input_width = input.size(-1)
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class IntelligentWaveletPoolingModule(nn.Module):
    def __init__(self, wavename='haar'):
        super(IntelligentWaveletPoolingModule, self).__init__()
        self.wavename = wavename
        self.dwt = DWT_2D(wavename=wavename)
        self.initialized = False  # 延迟初始化标志

    def build_layers(self, in_channels):
        self.in_channels = in_channels
        self.high_freq_conv = nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1)  # 保持通道不变
        self.softmax = nn.Softmax(dim=1)  # 注意：Softmax2d 已弃用，使用 dim=1 是对通道归一化
        self.fusion_conv = nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1)

    def forward(self, input):
        if not self.initialized:
            self.build_layers(input.size(1))
            self.high_freq_conv = self.high_freq_conv.to(input.device)
            self.fusion_conv = self.fusion_conv.to(input.device)
            self.initialized = True

        LL, LH, HL, HH = self.dwt(input)
        high_freq = torch.cat([LH, HL, HH], dim=1)  # (B, C*3, H/2, W/2)
        attention_map = self.softmax(self.high_freq_conv(high_freq))  # (B, C*3, H/2, W/2)
        enhanced_high_freq = high_freq * attention_map

        # 融合成 C 通道：对每组C通道分别求平均
        B, _, H, W = enhanced_high_freq.shape
        C = self.in_channels
        enhanced_chunks = torch.chunk(enhanced_high_freq, 3, dim=1)  # List of (B, C, H, W)
        high_freq_avg = sum(enhanced_chunks) / 3  # (B, C, H, W)

        combined_features = torch.cat([LL, high_freq_avg], dim=1)  # (B, 2C, H, W)
        output = self.fusion_conv(combined_features)  # (B, C, H, W)
        return output


if __name__ == "__main__":
    batch_size = 2
    channels = 32
    height, width = 256, 256
    input_tensor = torch.randn(batch_size, channels, height, width).cuda()
    iwpm = IntelligentWaveletPoolingModule().cuda()
    print(iwpm)
    print("\n哔哩哔哩：CV缝合救星\n")
    output = iwpm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
