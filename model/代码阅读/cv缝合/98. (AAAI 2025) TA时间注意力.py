import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
    
class TemporalAttention(nn.Module):
    def __init__(self, freq_num, channel, step, reduction=1, groups=1, select_method='all'):
        super(TemporalAttention, self).__init__()
        self.freq_num = freq_num
        self.channel = channel
        self.reduction = reduction
        self.select_method = select_method
        self.groups = groups
        self.step = step

        self.sigmoid = nn.Sigmoid()
        
        # cahnnel select
        self.avg_pool_c = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool_c = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.register_parameter('alpha', nn.Parameter(torch.FloatTensor([0.5])))
        self.register_parameter('beTemporalAttention', nn.Parameter(torch.FloatTensor([0.5])))

        
        # self.fc_c = nn.Linear(channel, channel, bias=False)
        self.fc_t = nn.Linear(step, step, bias=False)

        self.register_parameter('t', nn.Parameter(torch.FloatTensor([0.6])))    # m
        self.register_parameter('s', nn.Parameter(torch.FloatTensor([0.5])))  # n
        self.register_parameter('x', nn.Parameter(torch.FloatTensor([1])))

        self.register_parameter('t_scale', nn.Parameter(torch.FloatTensor([1])))
        self.register_parameter('s_scale', nn.Parameter(torch.FloatTensor([1])))

    def forward(self, x):
        t, b, c, h, w = x.shape
        x = rearrange(x, 't b c h w -> b t c h w')
        avg_map = self.avg_pool_c(x)    # (b, t, c, 1, 1)
        max_map = self.max_pool_c(x)

        map_add = self.alpha * avg_map + self.beTemporalAttention * max_map

        # time branch
        # map_fusion_t = self.fc_t(map_add)   # (b, t, c, 1, 1)
        map_add = rearrange(map_add, 'b t c 1 1 -> b c t')
        # map_fusion_t = self.fc_t(map_add.squeeze().transpose(1, 2)).transpose(1, 2) # (b, c, t) -> (b, t, c)
        map_fusion_t = self.fc_t(map_add).transpose(1, 2) # (b, c, t) -> (b, t, c)

        ## time
        t_mean_sig = self.sigmoid(torch.mean(map_fusion_t, dim=2))    # (b, t)
        t_mean_sig = rearrange(t_mean_sig, 'b t -> b t 1 1 1')
        t_mean_sig = t_mean_sig.repeat(1, 1, c, h, w)
        x_t = x * t_mean_sig + x    # (b, t, c, h, w)

        x = rearrange(x_t, 'b t c h w -> t b c h w')

        return x
    
if __name__ == "__main__":
    
    # 设置测试参数
    T = 2  # 时间步长
    B = 1  # 批次大小
    C = 32  # 通道数
    H = 256  # 高度
    W = 256  # 宽度
    # 创建一个随机输入张量，形状为 (T, B, C, H, W)
    x = torch.randn(T, B, C, H, W).cuda()
    # 初始化
    model = TemporalAttention(freq_num=9, channel=32, step=2, reduction=1, groups=1, select_method='all').cuda()
    print(model)
    # 运行模型前向传播
    output = model(x)
    print("\n哔哩哔哩: CV缝合救星!\n")
    # 打印输出的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
