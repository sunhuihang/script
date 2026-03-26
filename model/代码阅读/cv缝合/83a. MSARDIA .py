import torch
from torch import nn
import torch.nn.functional as F


class DeformableInteractiveAttention(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(DeformableInteractiveAttention, self).__init__()

        # 多尺度卷积层，用于提取不同尺度的特征
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if distortionmode:
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lra)

            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)

    def _adaptive_lr(self, feature):
        # 计算特征的标准差
        std = torch.std(feature)
        # 根据标准差动态调整学习率
        if std > 1.0:
            return 0.4
        elif std < 0.1:
            return 0.1
        else:
            return 0.2

    def _set_lra(self, module, grad_input, grad_output):
        lr = self._adaptive_lr(module.weight)
        grad_input = [g * lr if g is not None else None for g in grad_input]
        grad_output = [g * lr if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input

    def _set_lrm(self, module, grad_input, grad_output):
        lr = self._adaptive_lr(module.weight)
        grad_input = [g * lr if g is not None else None for g in grad_input]
        grad_output = [g * lr if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)

        out = torch.cat([max_out, avg_out], dim=1)

        if self.distortionmode:
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))
            d_max_out = torch.sigmoid(self.d_conv1(max_out))
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)

        # 多尺度特征提取
        out1 = self.conv1(out)
        out2 = self.conv2(out)
        out3 = self.conv3(out)
        # 融合多尺度特征
        out = out1 + out2 + out3

        mask = self.sigmoid(self.upsample(out))
        att_out = x * mask

        # 添加残差连接
        output = F.relu(att_out + x)
        return output


if __name__ == '__main__':
    B, C, H, W = 1, 32, 256, 256
    x = torch.randn(B, C, H, W).cuda()

    model = DeformableInteractiveAttention(stride=1, distortionmode=True).cuda()

    print(model)
    print("哔哩哔哩: CV缝合救星!")

    output = model(x)

    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")
