import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class DynamicChannelSplit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//4, 1),
            nn.GELU(),
            nn.Conv2d(dim//4, 4, 1),
        )

    def forward(self, x):
        weight = self.fc(x)  # (B,4,1,1)
        weight = F.softmax(weight.squeeze(-1).squeeze(-1), dim=1)  # (B,4)
        return weight

class SimpleAttentionBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return self.conv(x)

class DynamicLWGA_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.splitter = DynamicChannelSplit(dim)

        # Attention分支
        self.branch_pa = SimpleAttentionBranch(dim)
        self.branch_la = SimpleAttentionBranch(dim)
        self.branch_mra = SimpleAttentionBranch(dim)
        self.branch_ga = SimpleAttentionBranch(dim)

        # MLP融合
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1)
        )
        self.norm = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        weights = self.splitter(x)  # (B,4)
        B, C, H, W = x.shape

        # 将输入复制四份，每份乘以权重
        x1 = x * weights[:,0].view(B,1,1,1)
        x2 = x * weights[:,1].view(B,1,1,1)
        x3 = x * weights[:,2].view(B,1,1,1)
        x4 = x * weights[:,3].view(B,1,1,1)

        # 走各自分支
        x1 = self.branch_pa(x1)
        x2 = self.branch_la(x2)
        x3 = self.branch_mra(x3)
        x4 = self.branch_ga(x4)

        x_att = x1 + x2 + x3 + x4  # 融合（可以改成concat+conv也行）

        # MLP融合
        x = shortcut + self.drop_path(self.mlp(self.norm(x_att)))

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    dim = 32
    height, width = 20, 20
    input_tensor = torch.randn(batch_size, dim, height, width).to(device)

    model = DynamicLWGA_Block(dim=dim, mlp_ratio=4.0, drop_path=0.1).to(device)

    print(model)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("\n CV缝合救星：魔改版 Dynamic-LWGA 成功运行！")