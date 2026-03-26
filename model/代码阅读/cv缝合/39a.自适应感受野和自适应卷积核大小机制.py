import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


# 快速傅里叶变换（FFT）相关
def fft2d(x):
    # 对输入进行二维FFT
    return torch.fft.fft2(x)


def ifft2d(x):
    # 对输入进行二维逆FFT
    return torch.fft.ifft2(x)


# 频谱特征提取
def compute_spectrum(x):
    # 计算输入图像的频谱幅度
    fft_x = fft2d(x)
    magnitude = torch.abs(fft_x)  # 频谱的幅度
    return magnitude


# 自适应卷积核大小和感受野调整
class FADConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, fs_cfg=None):
        super(FADConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # 频谱选择配置
        if fs_cfg is not None:
            self.FS = FrequencySelection(in_channels, **fs_cfg)

        # 卷积层，初步使用固定卷积核
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation,
                              dilation=dilation)

    def adjust_kernel_and_dilation(self, x):
        # 计算频谱并根据频谱调整卷积核大小和扩张率
        spectrum = compute_spectrum(x)
        max_spectrum = spectrum.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # 频谱的最大值
        norm_spectrum = spectrum / (max_spectrum + 1e-6)  # 归一化频谱

        # 根据频谱信息动态调整扩张率
        dilation_factor = norm_spectrum.mean(dim=[-2, -1], keepdim=True).unsqueeze(1) * 2
        dilation = dilation_factor.mean()  # 改为取均值，确保 dilation 是标量

        # 确保 dilation 至少为 1
        dilation = max(int(dilation.item()), 1)

        # 根据频谱调整卷积核大小：高频区小卷积核，低频区大卷积核
        kernel_size_factor = 1 + 3 * (1 - norm_spectrum.mean(dim=[-2, -1], keepdim=True))  # 频谱较低时增大卷积核
        kernel_size = int(self.kernel_size * kernel_size_factor.mean().item())

        return kernel_size, dilation

    def forward(self, x):
        # 计算频谱并调整卷积核大小和扩张率
        kernel_size, dilation = self.adjust_kernel_and_dilation(x)

        # 动态调整卷积层的参数，确保 padding 和 dilation 为整数
        self.conv.kernel_size = (kernel_size, kernel_size)
        dilation = int(dilation)  # 确保 dilation 是整数
        self.conv.dilation = (dilation, dilation)
        self.conv.padding = (dilation, dilation)

        # 进行卷积操作
        x = self.conv(x)
        return x


# 自适应频谱选择模块
class FrequencySelection(nn.Module):
    def __init__(self, in_channels, k_list=[2], lp_type='avgpool', act='sigmoid', spatial_group=1):
        super().__init__()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.in_channels = in_channels
        self.spatial_group = spatial_group
        self.lp_type = lp_type

        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group

        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad2d(padding=k // 2),
                    nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
                ))

            for i in range(len(k_list)):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels,
                                             out_channels=self.spatial_group,
                                             stride=1,
                                             kernel_size=3,
                                             groups=self.spatial_group,
                                             padding=3 // 2,
                                             bias=True)
                self.freq_weight_conv_list.append(freq_weight_conv)

        self.act = act

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        return freq_weight

    def forward(self, x):
        x_list = []

        # Ensure correct processing for the frequency selection
        if self.lp_type == 'avgpool':
            pre_x = x
            b, _, h, w = x.shape
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](x)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
                                                                                               -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))

            x_list.append(pre_x)

        return x_list


# 测试代码
if __name__ == '__main__':
    input_tensor = torch.rand(1, 64, 64, 64)  # 输入形状 N C H W
    model = FADConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, fs_cfg={'k_list': [3, 5, 7]})
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
