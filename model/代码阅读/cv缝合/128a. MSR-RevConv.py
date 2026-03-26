import torch
import torch.nn as nn

class MSRRevConv(nn.Module):
    """
    Multi-Scale Regularized Reverse Convolution (MSR-RevConv)
    多尺度正则化逆卷积（魔改版）

    创新点：
    1）多尺度核（3x3 + 7x7）融合：适配细节恢复与模糊全局建模。
    2）通道自适应正则化λ：每个通道都有独立可学习的正则化强度。
    3）残差增强：在逆卷积输出基础上叠加上采样残差，保证细节稳定。
    """

    def __init__(self, in_channels, kernel_size=5, scale=1, padding=2, padding_mode='circular', eps=1e-5):
        super(MSRRevConv, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # 原始卷积核 (kernel_size)
        self.weight_main = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size))
        # 新增大核 (7x7)
        self.weight_large = nn.Parameter(torch.randn(1, in_channels, 7, 7))
        # 新增小核 (3x3)
        self.weight_small = nn.Parameter(torch.randn(1, in_channels, 3, 3))

        # 三个核的融合权重（Softmax保证非负且和为1）
        self.kernel_alpha = nn.Parameter(torch.ones(3))

        # 通道自适应正则化 λ
        self.lam = nn.Parameter(torch.full((1, in_channels, 1, 1), 1e-3))

        # 初始化 softmax 权重
        with torch.no_grad():
            self.weight_main.copy_(self._softmax_kernel(self.weight_main))
            self.weight_large.copy_(self._softmax_kernel(self.weight_large))
            self.weight_small.copy_(self._softmax_kernel(self.weight_small))

    def forward(self, x):
        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding],
                                  mode=self.padding_mode, value=0)

        _, _, h, w = x.shape

        # 上采样观测 Y_S
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x_up = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        else:
            x_up = x

        # 计算多尺度核 OTF
        FB_main = self.p2o(self.weight_main, (h * self.scale, w * self.scale))
        FB_large = self.p2o(self.weight_large, (h * self.scale, w * self.scale))
        FB_small = self.p2o(self.weight_small, (h * self.scale, w * self.scale))

        alphas = torch.softmax(self.kernel_alpha, dim=0)
        FB = alphas[0] * FB_main + alphas[1] * FB_large + alphas[2] * FB_small  # 融合核

        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))

        # 使用通道自适应 λ
        FR = FBFy + torch.fft.fftn(self.lam * x_up, dim=(-2, -1))
        x1 = FB.mul(FR)

        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + self.lam)

        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / self.lam
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding * self.scale:-self.padding * self.scale,
                      self.padding * self.scale:-self.padding * self.scale]

        # 残差增强（逆卷积结果 + 上采样输入）
        if out.shape == x_up.shape:
            out = out + x_up

        return out

    def splits(self, a, scale):
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale
        b = a.view(*leading_dims, scale, W_s, scale, H_s)
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1,
                                                          len(leading_dims) + 3,
                                                          len(leading_dims),
                                                          len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def p2o(self, psf, shape):
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2, -1))
        return otf

    def upsample(self, x, scale=3):
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)).type_as(x)
        z[..., ::scale, ::scale].copy_(x)
        return z

    @staticmethod
    def _softmax_kernel(k):
        B, C, H, W = k.shape
        k_ = k.view(B, C, -1)
        k_ = torch.softmax(k_, dim=-1)
        return k_.view(B, C, H, W)


if __name__ == "__main__":
    # 输入张量
    x = torch.randn(1, 32, 64, 64)

    # 初始化 MSR-RevConv
    model = MSRRevConv(in_channels=32, kernel_size=5, scale=2, padding=2)

    # 前向测试
    output = model(x)

    print(model)
    print("\n===== MSR-RevConv 测试 =====")
    print("输入形状:", x.shape)
    print("\n微信公众号:CV缝合救星\n")
    print("输出形状:", output.shape)
