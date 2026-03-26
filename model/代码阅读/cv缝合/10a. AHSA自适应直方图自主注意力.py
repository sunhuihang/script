import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

Conv2d = nn.Conv2d
"""
CV缝合救星原创魔改：Adaptive Histogram Self-Attention（AHSA）

局限性：
1. 排序操作的计算开销：每次执行排序操作时，会占用一定的计算资源，尤其是在处理大分辨率图像时，效率可能会受到影响。
改进方案可以通过减少排序操作的次数或优化排序方式来提升效率。
2. 基于直方图的自适应分组：现有实现对像素按强度进行排序后再分组，未能充分利用直方图的自适应性。引入直方图均衡或
自适应分组方式，以更好地适应不同的图像场景。

CV缝合救星改进方案：
1. 排序操作的计算开销：每次执行排序操作时，会占用一定的计算资源，尤其是在处理大分辨率图像时，效率可能会受到影响。
改进方案可以通过减少排序操作的次数或优化排序方式来提升效率。
2. 基于直方图的自适应分组：现有实现对像素按强度进行排序后再分组，未能充分利用直方图的自适应性。可以考虑引入直方图均
衡或自适应分组方式，以更好地适应不同的图像场景。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

Conv2d = nn.Conv2d


class ImprovedDHSA(nn.Module):
    def __init__(self, dim, num_heads=4, num_bins=4, bias=False, ifBox=True):
        super(ImprovedDHSA, self).__init__()
        self.factor = num_heads
        self.num_bins = num_bins
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)

    def adaptive_bin_sort(self, x, num_bins):
        # 自适应分组：基于像素强度的直方图分组
        bins = torch.linspace(x.min(), x.max(), num_bins + 1).to(x.device)
        indices = torch.bucketize(x, bins) - 1
        return indices, bins

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        # 自适应直方图分组
        intensity_map = x[:, :c // 2].view(b, -1)
        indices, bins = self.adaptive_bin_sort(intensity_map, self.num_bins)

        # 获取查询、键和值
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # BHR 和 FHR 的注意力计算
        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        # 将输出进行逐元素乘积融合
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2

        # 通过 1x1 卷积进行最终投影
        out = self.project_out(out)

        # 恢复直方图分组的顺序
        out_replace = out[:, :c // 2]

        # 调整 indices 使其与 out_replace 的维度相匹配
        indices = indices.view(b, c // 2, h, w)

        # 确保 indices 的最大值不超过 w-1 或 h-1，避免超出边界
        indices = indices.clamp(min=0, max=w - 1)

        out_replace = torch.scatter(out_replace, -1, indices, out_replace)
        out_replace = torch.scatter(out_replace, -2, indices, out_replace)

        out[:, :c // 2] = out_replace
        return out


# 测试模块
if __name__ == "__main__":
    model = ImprovedDHSA(64)
    input = torch.randn(1, 64, 128, 128)
    output = model(input)
    print('Input size:', input.size())
    print('Output size:', output.size())

