import torch
import torch.nn as nn
from torch.autograd import Function

# B站：CV缝合救星
"""
71. High-Similarity-Pass Attention for Single Image Super-Resolution（TIP 2024 顶刊）
    高相似度-通过注意力机制的单图像超分辨率 IEEE Transactions on Image Processing
    即插即用模块：High-Similarity-Pass Attention（HSPA）添花模块
    
一、背景
单图像超分辨率（SISR）旨在从低分辨率（LR）图像恢复出高分辨率（HR）图像，在多领域应用广泛。基于
深度学习的 SISR 常借助非局部注意力（NLA）探索图像自相似性，但 NLA 中的 softmax 变换在处理长
距离序列时存在问题，会使有价值的非局部信息重要性降低，引入无关特征干扰，导致 NLA 在探索图像自相
似性时效率低下。为解决该问题，提出高相似性通过注意力（HSPA）机制。

二、HSPA 介绍
（一）整体设计
HSPA 是一种用于 SISR 的注意力机制，通过引入软阈值（ST）操作改进 NLA，能够生成更紧凑、可解释的
概率分布，去除无关非局部信息，识别相关信息用于图像重建，可集成到现有深度 SISR 模型中，提升模型效
率和可解释性。
（二）核心组件与操作
1. 软阈值操作改进注意力权重分配：NLA 基于点积相似性融合非局部信息，但 softmax 变换使概率分布不合理。
2. HSPA 通过软阈值操作，根据相似性向量生成稀疏概率分布，将小概率值（低相似性）截断为零，为高相似性特
征向量分配更多权重。
3. 推导软阈值函数：对相似性向量排序，通过特定计算得到软阈值函数，确定截断阈值，使低于阈值的坐标截断为
零，其他坐标平移，保证概率值大于零且和为 1。
4. 计算 ST 操作导数：为实现 HSPA 的端到端训练，推导 ST 操作的封闭形式雅可比矩阵，其梯度计算复杂度低，
特殊结构使计算高效。

三、微观设计考量
从优化角度，HSPA 的注意力权重向量是原始相似性向量在单纯形上的投影，在 SISR 长序列建模中倾向于产生稀疏
概率分布。概率理论分析表明，随着非零元素数量期望增加，概率变小，符合设计初衷。实际应用中，可使用 top-k
替排序算法降低计算复杂度。

四、适用任务
1. 主要用于单图像超分辨率任务：在 SISR 任务中，HSPA 帮助模型更好地利用自相似性信息，对包含大量自相似信息
的数据集（如 Urban100 和 Manga109）效果显著。实验结果显示，HSPA 构建的 HSPAN 在定量和定性评估上均
优于现有方法，能修复严重受损纹理，恢复准确图像细节。
2. 可扩展到其他相关任务（分类、分割、检测等所有CV任务）：通过将 HSPA 集成到未使用 NLA 的 SISR 模型，以及
替换 NLA - 基于的深度 SISR 模型中的 softmax 变换，验证了 HSPA 的通用性，表明其可作为有效通用构建单元提
升多种模型性能。
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


if __name__ == "__main__":
    # 模块参数
    batch_size = 1  # 批大小
    channels = 256  # 输入特征通道数
    height = 32  # 图像高度
    width = 32  # 图像宽度
    # 创建 hspa 模块
    hspa = HSPA(channel=256, reduction=2, res_scale=1, conv=default_conv, topk=128)
    print(hspa)
    print("哔哩哔哩CV缝合救星, nice!")
    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = hspa(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)