import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
from typing import Optional

class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, proj_dim_in: Optional[int] = None):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            
            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, :self.proj_dim_in]).reshape(-1, 1, self.sk_size, self.sk_size)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size//2, groups=bs * self.pdim)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)
            
            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + x1_
            
            x = torch.cat([x1, x2], dim=1)
        else:
            dynamic_kernel = self.dwc_proj(x[:, :self.proj_dim_in]).reshape(-1, 1, self.sk_size, self.sk_size)
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + \
                rearrange(
                    F.conv2d(rearrange(x[:, :self.pdim], 'b c h w -> 1 (b c) h w'), dynamic_kernel, stride=1, padding=self.sk_size//2, groups=x.shape[0] * self.pdim), # 微信公众号:CV缝合救星
                    '1 (b c) h w -> b c h w', b=x.shape[0]
                )
        return x
    
    def extra_repr(self):
        return f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}'

if __name__ == "__main__":
    batch_size = 1
    channels = 32
    height = 256
    width = 256
    pdim = 32

    # 输入张量 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)

    # 生成一个随机的LK卷积核 根据原论文代码，形状为: [out_channels, in_channels, kernel_size, kernel_size]
    # 源代码中kernel_size=13, 大家可以设置为3等，测试实验效果
    lk_filter = torch.randn(pdim, pdim, 13, 13)

    # 实例化模型
    model = ConvolutionalAttention(pdim=pdim)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    lk_filter = lk_filter.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x, lk_filter)

    # 输出模型结构与形状信息
    print(model)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]