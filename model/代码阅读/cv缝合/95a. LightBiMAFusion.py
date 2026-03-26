import torch
import torch.nn as nn
import torch.nn.functional as F


class LightBiMAFusion(nn.Module):
    """
    LightBiMAFusion: 轻量化双向模态融合模块
    - 高分辨率输入支持（如 256×256）
    - 注意力仅在低分辨率上计算，节省显存
    - 支持图像语义双向融合
    """
    def __init__(self, img_channels, txt_channels, mid_channels=64, attn_size=64):
        super(LightBiMAFusion, self).__init__()
        self.attn_size = attn_size

        self.img_proj = nn.Conv2d(img_channels, mid_channels, kernel_size=1)
        self.txt_proj = nn.Conv2d(txt_channels, mid_channels, kernel_size=1)

        self.gamma_img2txt = nn.Parameter(torch.zeros(1))
        self.gamma_txt2img = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_feat, txt_feat):
        B, _, H, W = img_feat.size()
        target_size = (H, W)

        # 下采样做注意力
        img_down = F.adaptive_avg_pool2d(img_feat, (self.attn_size, self.attn_size))
        txt_down = F.adaptive_avg_pool2d(txt_feat, (self.attn_size, self.attn_size))

        # 通道映射
        img_proj = self.img_proj(img_down)
        txt_proj = self.txt_proj(txt_down)

        B, C, h, w = img_proj.shape
        N = h * w

        # reshape
        Q_txt = txt_proj.view(B, C, N)
        K_img = img_proj.view(B, C, N)
        V_img = img_proj.view(B, C, N).permute(0, 2, 1)
        V_txt = txt_proj.view(B, C, N).permute(0, 2, 1)

        # 注意力矩阵
        attn_img2txt = self.softmax(torch.bmm(Q_txt.permute(0, 2, 1), K_img))
        attn_txt2img = self.softmax(torch.bmm(K_img.permute(0, 2, 1), Q_txt))

        # 注意力融合
        fusion_img = torch.bmm(attn_img2txt, V_img).permute(0, 2, 1).view(B, C, h, w)
        fusion_txt = torch.bmm(attn_txt2img, V_txt).permute(0, 2, 1).view(B, C, h, w)

        # 上采样恢复
        fusion_img_up = F.interpolate(fusion_img, size=target_size, mode='bilinear', align_corners=False)
        fusion_txt_up = F.interpolate(fusion_txt, size=target_size, mode='bilinear', align_corners=False)

        # 残差融合
        out_img = self.gamma_img2txt * fusion_img_up + self.txt_proj(txt_feat)
        out_txt = self.gamma_txt2img * fusion_txt_up + self.img_proj(img_feat)

        return out_img, out_txt


# -------------------- 测试代码 --------------------
if __name__ == "__main__":
    B, C_img, C_txt = 1, 3, 3
    H, W = 256, 256

    img_input = torch.randn(B, C_img, H, W).cuda()
    txt_input = torch.randn(B, C_txt, H, W).cuda()

    model = LightBiMAFusion(img_channels=C_img, txt_channels=C_txt, mid_channels=64, attn_size=64).cuda()
    out_img, out_txt = model(img_input, txt_input)

    print("\n 哔哩哔哩CV缝合救星原创魔改")
    print("--------- 输入特征 ---------")
    print(f"输入图像特征 shape : {img_input.shape}")
    print(f"输入语义特征 shape : {txt_input.shape}")

    print("--------- 输出特征 ---------")
    print(f"融合后图像特征 shape : {out_img.shape}")
    print(f"融合后语义特征 shape : {out_txt.shape}")
