import torch
import torch.nn as nn
import torch.nn.functional as F


class GCABlock(nn.Module):
    """
    GCABlock: Gated Channel Attention Block（魔改版 SCA）
    功能特点：
        - 通道压缩与扩展（Bottleneck）
        - 通道注意力评分 + 门控机制
        - 可配置残差连接方式：'add' 或 'concat'

    参数：
        c : 输入/输出通道数
        reduction : 压缩倍率（默认 2）
        residual_mode : 残差连接方式，'add' 或 'concat'
    """

    def __init__(self, c, reduction=2, residual_mode='add'):
        super().__init__()
        self.c = c
        self.reduction = reduction
        self.residual_mode = residual_mode

        mid_c = max(1, c // reduction)

        # 通道特征生成
        self.fq = nn.Sequential(
            nn.Conv2d(c, mid_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, c, kernel_size=3, padding=1, bias=False)
        )
        self.fk = nn.Sequential(
            nn.Conv2d(c, mid_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, c, kernel_size=3, padding=1, bias=False)
        )
        self.fv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)

        # 门控机制
        self.sigmoid_gate = nn.Sigmoid()

        # 如果是 concat 模式，需要降维和匹配 BatchNorm
        if residual_mode == 'concat':
            self.reproject = nn.Conv2d(c * 2, c, kernel_size=1)
            self.bn = nn.BatchNorm2d(c)
        elif residual_mode == 'add':
            self.bn = nn.BatchNorm2d(c)
        else:
            raise ValueError("residual_mode must be 'add' or 'concat'")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        # 获取 query、key、value 特征
        fq = self.fq(inputs)  # [B,C,H,W]
        fk = self.fk(inputs)
        fv = self.fv(inputs)

        # 注意力打分
        f_sim = torch.matmul(fq, fk.transpose(2, 3)) / (fq.size(-1) ** 0.5)
        f_sum = torch.sum(f_sim, dim=(2, 3))  # [B,C]
        softmax_score = torch.softmax(f_sum, dim=1).unsqueeze(2).unsqueeze(3)  # [B,C,1,1]
        gate_score = self.sigmoid_gate(f_sum).unsqueeze(2).unsqueeze(3)

        # 权重融合
        score = softmax_score * gate_score
        out = score * fv

        # 残差连接方式
        if self.residual_mode == 'add':
            r = out + inputs
        else:  # concat 模式
            r = torch.cat([out, inputs], dim=1)
            r = self.reproject(r)

        r = self.bn(r)
        r = self.relu(r)
        return r


# 测试模块运行情况
if __name__ == "__main__":
    batch_size = 1
    channels = 32
    height = 256
    width = 256

    x = torch.randn(batch_size, channels, height, width)
    model = GCABlock(c=channels, reduction=2, residual_mode='concat')  # 或 'add'

    print(model)
    print("哔哩哔哩：CV缝合救星!")
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
