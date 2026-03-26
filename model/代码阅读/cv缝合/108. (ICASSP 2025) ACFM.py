import torch
import torch.nn as nn
from einops import rearrange

# CV缝合救星2025.06.08视频
# 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
class ACFMAttention(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(ACFMAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
 
        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)
 
        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)

    # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)
 
        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)
 
 
        # global SA
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
 
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
 
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
 
        out = (attn @ v)
        # 请支持正版（倒卖盗版者我已掌握你的ip和信息，请好自为之；使用盗版者，发文运气还是需要积累的，请支持正版。）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output =  out + out_conv
 
        return output

if __name__ == "__main__":
     # 模拟输入参数
    batch_size = 1
    channels = 32
    height = 256
    width = 256
    num_heads = 4

    # 创建输入张量：形状 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)

    model = ACFMAttention(dim=channels, num_heads=num_heads, bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x)
    print(model)
    print("哔哩哔哩:CV缝合救星")

    # 打印输入输出形状
    print("输入形状:", x.shape)     # [B, C, H, W]
    print("输出形状:", output.shape)  # [B, C, H, W]
