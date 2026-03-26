import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor
import torch.nn.functional as F


# PA Block: Point Attention
class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            norm_layer(dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)
        return x


# LA Block: Local Attention
class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Linear Attention Block
class LinearAttention(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = norm_layer(dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Compute Query, Key, Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Flatten spatial dimensions for linear attention
        B, C, H, W = x.size()
        Q = Q.view(B, C, -1)
        K = K.view(B, C, -1)
        V = V.view(B, C, -1)

        # Linear attention approximation
        attention = torch.bmm(Q.permute(0, 2, 1), K)
        attention = self.softmax(attention)  # Apply softmax to normalize attention weights
        out = torch.bmm(attention, V.permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(B, C, H, W)

        # Apply normalization after attention
        return self.norm(out)


# Strip Convolution Blocks
class StripConv(nn.Module):
    def __init__(self, dim, kernel_size, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = norm_layer(dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# LWGA Block with Channel Split and Fusion
class LWGA_Block(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 5  # Split the channels into 5 parts
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        # PA, LA, Linear Global Attention, and Strip Convolutions
        self.PA = PA(self.dim_split, norm_layer, act_layer)  # PA is point attention
        self.LA = LA(self.dim_split, norm_layer, act_layer)  # LA is local attention
        self.GlobalAttn = LinearAttention(self.dim_split, norm_layer)  # Linear Global Attention

        self.strip_conv_1x11 = StripConv(self.dim_split, 1, norm_layer, act_layer)  # 1x11 strip convolution
        self.strip_conv_11x1 = StripConv(self.dim_split, 11, norm_layer, act_layer)  # 11x1 strip convolution

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x.clone()

        # Dynamically calculate split size based on number of channels
        channels = x.size(1)
        split_size = channels // 5

        if channels % 5 != 0:
            raise ValueError("The number of input channels must be divisible by 5.")

        # Split channels into 5 parts
        x_parts = torch.split(x, [split_size] * 5, dim=1)

        # Process each part separately through PA, LA, Linear Global Attention, and Strip Convolutions
        x1 = self.PA(x_parts[0])
        x2 = self.LA(x_parts[1])
        x3 = self.GlobalAttn(x_parts[2])

        x4 = self.strip_conv_1x11(x_parts[3])
        x5 = self.strip_conv_11x1(x_parts[4])

        # Concatenate the processed parts
        x_att = torch.cat([x1, x2, x3, x4, x5], dim=1)

        # Apply MLP and skip connection
        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))

        return x


if __name__ == "__main__":
    # Test the modified LWGA_Block
    batch_size = 1
    dim = 30  # Adjust the number of channels to be divisible by 5
    height, width = 20, 20
    input_tensor = torch.randn(batch_size, dim, height, width).cuda()

    # Initialize LWGA_Block
    model = LWGA_Block(
        dim=dim,
        stage=2,
        att_kernel=3,
        mlp_ratio=4.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d
    ).cuda()

    print(model)
    print("\n 哔哩哔哩：CV缝合救星!\n")

    # Forward pass
    output = model(input_tensor)

    # Print input and output shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
