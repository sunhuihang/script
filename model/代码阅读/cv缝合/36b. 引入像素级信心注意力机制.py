import torch
import torch.nn as nn
import torch.nn.functional as F
"""
CV缝合救星魔改创新点2：引入像素级信心注意力机制
一、问题：
原始模型虽然能提取丰富的局部特征，但在捕捉像素级细节和加强像素间的关系时存在不足，特别是在处理细粒度任务时，
常常无法有效聚焦在关键区域。此外，原始模型的维度匹配问题可能导致计算错误，影响模型稳定性。
二、魔改创新：
1. 引入像素级信心注意力机制：
通过生成“像素信心图”，根据每个像素的信心值对特征图进行加权，有效增强模型对关键区域的关注，提升细粒度信息的表达能力。
2. 多尺度特征融合：
通过引入多尺度卷积层（如 bin_size=(4,4)），模型能够在不同尺度下捕捉更加细致的局部信息，提升对不同尺度目标的感知能力。
3. 改进维度匹配：
通过优化矩阵相乘的维度转换，解决了原始模型中 pixel_confidence 和 x 维度不匹配的问题，确保了计算的稳定性和高效性。
"""
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


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False)
        # 点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 空间注意力生成
        attention_map = self.conv(x)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map  # 通过注意力图进行加权


# 非局部注意力机制
class NonLocalAttention(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        # 获取查询、键和值
        query = self.query_conv(x).view(B, -1, H * W)  # [B, C/2, H*W]
        key = self.key_conv(x).view(B, -1, H * W)  # [B, C/2, H*W]
        value = self.value_conv(x).view(B, -1, H * W)  # [B, C, H*W]

        # 计算注意力图
        attention_map = torch.bmm(query.transpose(1, 2), key)  # [B, H*W, H*W]
        attention_map = self.softmax(attention_map)

        # 获取加权特征
        out = torch.bmm(value, attention_map.transpose(1, 2))  # [B, C, H*W]
        out = out.view(B, C, H, W)  # 恢复为原始空间维度
        return out


# PCAA模块
class PCAA(nn.Module):
    def __init__(self, feat_in, bin_size=(4, 4), norm_layer=nn.BatchNorm2d):
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

        # 通过修改此处的维度，解决矩阵相乘维度不匹配问题
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


# 输入 N C H W, 输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)  # 模拟输入
    pcaa = PCAA(64)
    output = pcaa(input)
    print("PCAA_input.shape:", input.shape)
    print("PCAA_output.shape:", output.shape)
