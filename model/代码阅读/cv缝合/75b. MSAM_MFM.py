import torch
import torch.nn as nn
import torch.nn.functional as F


class MSAM_MFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(MSAM_MFM, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)

        # 多尺度卷积
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)

        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.mlp4 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats1, in_feats2):
        in_feats = [in_feats1, in_feats2]
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        # 多尺度特征融合
        feats_sum1 = torch.sum(in_feats, dim=1)
        feats_sum2 = F.interpolate(feats_sum1, scale_factor=0.5, mode='bilinear', align_corners=False)
        feats_sum4 = F.interpolate(feats_sum1, scale_factor=0.25, mode='bilinear', align_corners=False)

        attn1 = self.mlp1(self.avg_pool1(feats_sum1))
        attn2 = self.mlp2(self.avg_pool2(feats_sum2))
        attn4 = self.mlp4(self.avg_pool4(feats_sum4))

        # 确保attn1维度正确，添加维度扩展
        if attn1.ndim == 2:
            attn1 = attn1.unsqueeze(2).unsqueeze(3)
        elif attn1.ndim == 3:
            attn1 = attn1.unsqueeze(3)

        # 修改此处，size 参数只设置空间维度
        attn1 = F.interpolate(attn1, size=(1, 1), mode='bilinear', align_corners=False)

        # 确保attn2维度正确，添加维度扩展
        if attn2.ndim == 2:
            attn2 = attn2.unsqueeze(2).unsqueeze(3)
        elif attn2.ndim == 3:
            attn2 = attn2.unsqueeze(3)

        # 修改此处，size 参数只设置空间维度
        attn2 = F.interpolate(attn2, size=(1, 1), mode='bilinear', align_corners=False)

        # 确保attn4维度正确，添加维度扩展
        if attn4.ndim == 2:
            attn4 = attn4.unsqueeze(2).unsqueeze(3)
        elif attn4.ndim == 3:
            attn4 = attn4.unsqueeze(3)

        # 修改此处，size 参数只设置空间维度
        attn4 = F.interpolate(attn4, size=(1, 1), mode='bilinear', align_corners=False)

        attn = attn1 + attn2 + attn4
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


if __name__ == "__main__":
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)
    mfm = MSAM_MFM(64)
    output = mfm(input1, input2)
    print("MSAM_MFM_输入张量形状:", input1.shape)
    print("MSAM_MFM_输出张量形状:", output.shape)