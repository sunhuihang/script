import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class F2A(nn.Module):
    """
    Frequency-Filtered Attention (F2A)
    输入 (B, C, H, W)，输出 (B, C, H, W)
    """
    def __init__(self, freq_keep_ratio=0.5):
        super(F2A, self).__init__()
        self.freq_keep_ratio = freq_keep_ratio
        self.scale = nn.Parameter(torch.ones(1))  # 可学习缩放因子
        self.bias = nn.Parameter(torch.zeros(1))  # 可学习偏置项

    def forward(self, x):
        b, c, h, w = x.shape

        # 将输入进行2D DCT变换（采用fft实现近似）
        x_fft = torch.fft.fft2(x, norm='ortho')  # 复杂数
        x_fft_amp = torch.abs(x_fft)  # 取幅值

        # 频率掩码：只保留能量最高的前freq_keep_ratio比例
        flat = x_fft_amp.view(b, c, -1)
        threshold_idx = int(flat.size(-1) * self.freq_keep_ratio)
        threshold_values, _ = torch.topk(flat, threshold_idx, dim=-1)
        min_threshold = threshold_values.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        freq_mask = (x_fft_amp >= min_threshold.view(b, c, 1, 1)).float()

        # 抑制低能量频率
        x_fft_filtered = x_fft * freq_mask

        # 反变换回空间域
        x_filtered = torch.fft.ifft2(x_fft_filtered, norm='ortho').real

        # 残差连接并加上可学习缩放与偏置
        out = self.scale * x_filtered + self.bias + x

        return out


if __name__ == "__main__":
    # 测试代码
    B, C, H, W = 2, 32, 64, 64
    x = torch.randn(B, C, H, W).cuda()
    model = F2A(freq_keep_ratio=0.5).cuda()
    print(model)
    output = model(x)
    print("\n 哔哩哔哩：CV缝合救星")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")