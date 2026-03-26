import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
"""
CV缝合救星魔改创新：引入自适应感受野和自适应卷积核大小机制
一、背景
1. 自适应感受野：感受野的大小会影响网络的特征提取能力。在一些任务中，可能需要局部细节特征，
而在另一些任务中则可能需要更广泛的上下文信息。因此，自适应感受野的机制可以根据输入的内容
决定感受野的大小。
2. 自适应卷积核大小：不同区域的特征可能需要不同大小的卷积核进行处理。比如，物体的边缘特征
需要较小的卷积核，而背景区域则适合使用较大的卷积核。通过引入自适应卷积核大小，可以使网络根
据输入的局部特征信息，动态选择最合适的卷积核大小，从而提升模型的灵活性。
二、实现思路：
1. 自适应感受野：通过动态调整卷积操作的扩张率（dilation）来改变感受野。
2. 自适应卷积核大小：在 forward 方法中，通过对输入图像的频谱分析来动态选择不同的卷积核大小。
三、关键修改：
1.卷积核大小动态调整：基于输入特征图的频谱信息（使用 FFT 变换），在高频区域使用小的卷积核，
在低频区域使用大的卷积核。
2. 扩张率调整：通过计算特征图的频谱，动态调整扩张率，以实现不同区域不同感受野的自适应调整。
"""
try:
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
except ImportError as e:
    ModulatedDeformConv2d = nn.Module


class OmniAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


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
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))

            x_list.append(pre_x)

        return x_list


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

        # 根据频谱调整卷积核大小：高频区小卷积核，低频区大卷积核
        kernel_size_factor = 1 + 3 * (1 - norm_spectrum.mean(dim=[-2, -1], keepdim=True))  # 频谱较低时增大卷积核
        kernel_size = int(self.kernel_size * kernel_size_factor.mean().item())

        return kernel_size, dilation

    def forward(self, x):
        # 计算频谱并调整卷积核大小和扩张率
        kernel_size, dilation = self.adjust_kernel_and_dilation(x)

        # 动态调整卷积层的参数
        self.conv.kernel_size = (kernel_size, kernel_size)
        self.conv.dilation = (dilation.item(), dilation.item())  # 确保 dilation 是标量
        self.conv.padding = (dilation.item(), dilation.item())

        # 进行卷积操作
        x = self.conv(x)
        return x


# 测试代码
if __name__ == '__main__':
    input_tensor = torch.rand(1, 64, 64, 64)  # 输入形状 N C H W
    model = FADConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, fs_cfg={'k_list': [3, 5, 7]})
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
