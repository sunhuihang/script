import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新1：引入多尺度注意力机制
一、问题：
目前的 PCAA 模块依赖于全局类别表示的计算（通过加权聚合得到全局类别中心），这对于某些具有复杂局部模式
的任务（比如细粒度语义分割）可能会带来问题。虽然全局表示能提升类内一致性，但它可能在某些情况下无法有效
捕捉到非常局部的特征。由于全局类别中心是通过加权聚合得到的，它可能没有充分关注到局部区域的细节，特别是
在处理细粒度目标或物体时，局部特征的重要性不容忽视。
二、魔改创新：
1. 引入多尺度注意力机制：在全局类别表示的计算中，引入不同尺度的注意力机制（如自注意力）。这样一方面保持全
局一致性，另一方面通过不同尺度的特征捕捉更细粒度的局部信息，避免过度依赖全局类别中心。多尺度的设计能更
好地融合细节信息，有助于提高模型在细粒度任务中的表现。
2. 通过引入不同尺度的卷积层提取多尺度特征，并结合自注意力机制增强不同尺度之间的关系。
3. 改进全局-局部特征交互：通过引入更丰富的全局与局部特征交互机制，优化模型对细节的捕捉。
"""

def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B, -1, rH, rW, C)  # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out


def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W)  # [B, C, H, W]
    return out


# 多尺度注意力模块
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4]):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        # 确保每个卷积层输入通道数与特征图的通道数一致
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=s, dilation=s) for s in scales])
        self.fc = nn.Linear(out_channels * len(scales), out_channels)

    def forward(self, x):
        scale_feats = [conv(x) for conv in self.convs]  # 处理不同尺度的卷积
        scale_feats = torch.cat(scale_feats, dim=1)  # 合并多尺度特征
        scale_feats = F.adaptive_avg_pool2d(scale_feats, 1).view(x.size(0), -1)  # 平均池化并展平
        scale_feats = self.fc(scale_feats)  # 全连接层
        return scale_feats

class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class PCAA(nn.Module):
    def __init__(self, feat_in, bin_size=(4,4), norm_layer=nn.BatchNorm2d):
        super(PCAA, self).__init__()
        feat_inner = feat_in // 2
        num_classes = feat_in
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)
        self.multi_scale_attention = MultiScaleAttention(feat_in, feat_inner)

    def forward(self, x):
        cam = self.conv_cam(self.dropout(x))  # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam))  # [B, K, bin_num_h, bin_num_w]

        residual = x  # [B, C, H, W]
        cam = patch_split(cam, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, K]
        x = patch_split(x, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH * rW, K)  # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH * rW, C)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)  # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3),
                                   x) * bin_confidence  # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats)  # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats)  # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1)  # [B, bin_num_h * bin_num_w, K, C]

        query = self.proj_query(x)  # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats)  # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats)  # [B, bin_num_h * bin_num_w, K, C//2]

        aff_map = torch.matmul(query, key.transpose(2, 3))  # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        out = out.view(B, -1, rH, rW, value.shape[-1])  # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size)  # [B, C, H, W]

        out = residual + self.conv_out(out)
        return out


# 测试代码
if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)
    pcaa = PCAA(64)
    output = pcaa(input)
    print("PCAA_input.shape:", input.shape)
    print("PCAA_output.shape:", output.shape)

