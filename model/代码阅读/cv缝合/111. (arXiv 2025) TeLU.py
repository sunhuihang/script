import torch

def TeLU(input):
  return input * torch.tanh( torch.exp(input) )

if __name__ == "__main__":

    # 创建输入张量：形状 [B, C, H, W]
    x = torch.randn(1, 3, 256, 256).cuda()

    # 调用TeLU方法
    output = TeLU(x).cuda()

    print("\nTeLU: \n哔哩哔哩：CV缝合救星！\n")

    # 打印输入输出形状
    print("输入形状:", x.shape)     # [B, C, H, W]
    print("输出形状:", output.shape)  # [B, C_out, H_out, W_out]
