import torch
import torch.nn as nn
from torch.autograd import Function

"""
CV缝合救星魔改创新：多尺度HSPA模块（Multi-Scale HSPA, MS-HSPA）
一、创新点思路：
1. 在原始的 HSPA 模块基础上，引入多尺度特征融合的思想。原始的 HSPA 主要在单一尺度上进行
注意力计算，而多尺度特征融合可以让模型在不同尺度下捕捉图像的特征信息，进一步增强模型对图像自
相似性的挖掘能力，提升单图像超分辨率的效果。
2. 通过不同卷积核大小的卷积层来提取不同尺度的特征，然后分别对这些特征应用 HSPA 操作，最后将
不同尺度的结果进行融合。

二、代码解释
1. MultiScaleHSPA 类：这是新构建的多尺度 HSPA 模块。在初始化时，定义了不同卷积核大小（3x3、
5x5、7x7）的卷积层用于提取不同尺度的特征，以及对应的 HSPA 模块对这些特征进行处理。最后，使用
一个 1x1 卷积层将不同尺度的输出特征进行融合。
2. forward 方法：首先通过不同的卷积层提取不同尺度的特征，然后分别对这些特征应用 HSPA 操作，
得到不同尺度的输出。接着，将这些输出特征在通道维度上进行拼接，最后通过融合卷积层得到最终的输出。
"""

class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


# High-Similarity-Pass Attention
class HSPA(nn.Module):
    def __init__(self, channel=256, reduction=2, res_scale=1, conv=default_conv, topk=128):
        super(HSPA, self).__init__()
        self.res_scale = res_scale
        self.conv_match1 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        self.ST = SoftThresholdingOperation(dim=2, topk=topk)

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)

        score = torch.matmul(x_embed_1, x_embed_2)
        score = self.ST(score)

        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        return self.res_scale * x_final.permute(0, 2, 1).view(N, -1, H, W) + input


class SoftThresholdingOperation(nn.Module):
    def __init__(self, dim=2, topk=128):
        super(SoftThresholdingOperation, self).__init__()
        self.dim = dim
        self.topk = topk

    def forward(self, x):
        return softThresholdingOperation(x, self.dim, self.topk)


def softThresholdingOperation(x, dim=2, topk=128):
    return SoftThresholdingOperationFun.apply(x, dim, topk)


class SoftThresholdingOperationFun(Function):
    @classmethod
    def forward(cls, ctx, s, dim=2, topk=128):
        ctx.dim = dim
        max, _ = s.max(dim=dim, keepdim=True)
        s = s - max
        tau, supp_size = tau_support(s, dim=dim, topk=topk)
        output = torch.clamp(s - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


def tau_support(s, dim=2, topk=128):
    if topk is None or topk >= s.shape[dim]:
        k, _ = torch.sort(s, dim=dim, descending=True)
    else:
        k, _ = torch.topk(s, k=topk, dim=dim)

    topk_cumsum = k.cumsum(dim) - 1
    ar_x = ix_like_fun(k, dim)
    support = ar_x * k > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(s.dtype)

    if topk is not None and topk < s.shape[dim]:
        unsolved = (support_size == topk).squeeze(dim)

        if torch.any(unsolved):
            in_1 = roll_fun(s, dim)[unsolved]
            tau_1, support_size_1 = tau_support(in_1, dim=-1, topk=2 * topk)
            roll_fun(tau, dim)[unsolved] = tau_1
            roll_fun(support_size, dim)[unsolved] = support_size_1

    return tau, support_size


def ix_like_fun(x, dim):
    d = x.size(dim)
    ar_x = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return ar_x.view(view).transpose(0, dim)


def roll_fun(x, dim):
    if dim == -1:
        return x
    elif dim < 0:
        dim = x.dim() - dim

    perm = [i for i in range(x.dim()) if i != dim] + [dim]
    return x.permute(perm)


# 多尺度 HSPA 模块
class MultiScaleHSPA(nn.Module):
    def __init__(self, channel=256, reduction=2, res_scale=1, conv=default_conv, topk=128):
        super(MultiScaleHSPA, self).__init__()
        self.res_scale = res_scale
        # 不同尺度的卷积层
        self.conv_3x3 = BasicBlock(conv, channel, channel, 3, bn=False, act=nn.PReLU())
        self.conv_5x5 = BasicBlock(conv, channel, channel, 5, bn=False, act=nn.PReLU())
        self.conv_7x7 = BasicBlock(conv, channel, channel, 7, bn=False, act=nn.PReLU())
        # 不同尺度的 HSPA 模块
        self.hspa_3x3 = HSPA(channel=channel, reduction=reduction, res_scale=res_scale, conv=conv, topk=topk)
        self.hspa_5x5 = HSPA(channel=channel, reduction=reduction, res_scale=res_scale, conv=conv, topk=topk)
        self.hspa_7x7 = HSPA(channel=channel, reduction=reduction, res_scale=res_scale, conv=conv, topk=topk)
        # 融合卷积层
        self.fusion_conv = BasicBlock(conv, 3 * channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        # 不同尺度的特征提取
        feat_3x3 = self.conv_3x3(input)
        feat_5x5 = self.conv_5x5(input)
        feat_7x7 = self.conv_7x7(input)
        # 不同尺度的 HSPA 操作
        out_3x3 = self.hspa_3x3(feat_3x3)
        out_5x5 = self.hspa_5x5(feat_5x5)
        out_7x7 = self.hspa_7x7(feat_7x7)
        # 特征融合
        fused_feat = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)
        final_output = self.fusion_conv(fused_feat)
        return final_output + input


if __name__ == "__main__":
    # 模块参数
    batch_size = 1  # 批大小
    channels = 256  # 输入特征通道数
    height = 32  # 图像高度
    width = 32  # 图像宽度
    # 创建 MS-HSPA 模块
    ms_hspa = MultiScaleHSPA(channel=256, reduction=2, res_scale=1, conv=default_conv, topk=128)
    print(ms_hspa)
    print("哔哩哔哩CV缝合救星, nice!")
    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = ms_hspa(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)