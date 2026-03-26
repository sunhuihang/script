import torch.nn as nn
from torch.nn import functional as F
import torch
"""
部分类别激活注意力（PCAA）模块：用于语义分割任务的注意力模块（CVPR 2022）
即插即用模块：PCAA（可集成到多种基于 CNN 的网络中，增强特征表示，提升分割性能）
一、背景
在语义分割任务中，为实现精确分割，需消除像素因纹理、光照和位置差异导致的局部特异性，生成具有全局一致性的特征。
当前基于注意力的方法主要通过成对亲和力和粗分割来建模像素关系，但非局部模型计算密集且像素级聚合无法保证全局一
致性，基于粗分割的模型则忽略了局部特异性。为解决这些问题，本文提出 PCAA 模块，旨在引入新的像素关系建模方式，
同时考虑局部特异性和全局一致性。
二、PCAA模块原理
1. 输入特征：来自卷积网络的特征，其通道数可先通过 3×3 卷积块减少到 512，然后输入PCAA模块。
2. 模块组成与处理：
A. 部分类别激活图（Partial CAM）：为解决传统 CAM 在空间信息利用上的不足，用自适应平均池化代替全局平均池化，
将图像划分为非重叠的块，在每个块内进行区域级预测生成 Partial CAM。这使得网络能学习更多空间信息，提供更精确
的定位，其预测可视为每个块内的多标签分类任务，且能根据像素级注释计算出更精细的监督标签。
B. 局部类别中心计算：根据生成的 Partial CAM，通过加权求和计算每个块内的局部类别中心，这些局部表示能更好地
体现不同局部上下文下的特征。为提高局部中心的语义代表性，采用图卷积单元（GCU）构建局部中心之间的交互，更新节点特征。
C. 全局类别表示获取：将所有区域的局部中心通过加权聚合得到全局类别中心，以提高整幅图像的类内一致性。
D. 特征聚合：利用局部和全局类别中心进行注意力计算，先通过局部中心计算每个区域内的像素相似度图 ，再用全局中心
聚合增强特征得到 ，最后恢复到原始尺寸作为注意力计算的输出。
3. 输出融合与增强：PCAA 模块的输出增强特征与输入特征拼接，生成最终的分割图。
三、适用任务
该模块适用于图像分类、目标检测、语义分割等所有计算机视觉任务。
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

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)
    pcaa = PCAA(64)
    output = pcaa(input)
    print("PCAA_input.shape:", input.shape)
    print("PCAA_output.shape:",output.shape)




