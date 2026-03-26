import torch
import torch.nn as nn

# 哔哩哔哩：CV缝合救星
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class DyTConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, shape,padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1,shape=[], p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.DyT = DynamicTanh(shape)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, DynamicTanh  and activation to input tensor."""
        x = self.conv(x)
        out = self.bn(x*self.DyT(x)) #大家可以合理玩一下这个DyT模块，但是不要直接替换bn批标准化，不然容易造成训练不稳定。
        return self.act(out)

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    input = torch.randn(1,32,128, 128)  # 创建一个形状为 (1,32,128, 128)
    DyT = DynamicTanh([32,128,128])
    output = DyT(input)  # 通过 DyTConv 模块计算输出
    print('DyT_Input size:', input.size())  # 打印输入张量的形状
    print('DyT_Output size:', output.size())  # 打印输出张量的形状


    input_tensor = torch.randn(1,32,128, 128)  # 创建一个形状为 (1,32,128, 128)
    # 创建 DyTConv 模块实例，输入通道数为32，输出通道数为 64，卷积核为1，步长为1。
    module =DyTConv(32,64,3,2,[64,64,64])
    output_tensor = module(input_tensor)  # 通过 DyTConv 模块计算输出
    print("哔哩哔哩: CV缝合救星!")
    print('DyTConv_Input size:', input_tensor.size())  # 打印输入张量的形状
    print('DyTConv_Output size:', output_tensor.size())  # 打印输出张量的形状