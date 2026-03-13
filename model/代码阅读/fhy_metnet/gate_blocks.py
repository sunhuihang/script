import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint
from model.attn_layers import Attention2D_ROPE
from model.norm_layers import RMSNorm, ChanLayerNorm, DyT,DyT_4D
from einops import rearrange

try:
    from .utils import apply_initialization
except:
    from utils import apply_initialization

class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features=None,
            out_features=None,
            bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SpaceTransformerBlock(nn.Module):
    """
       A DiT block with `add` conditioning.
       """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 input_size,
                 mlp_ratio=4.0,
                 qk_norm=False,
                 norm_type="layer",
                 use_checkpoint=False
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if norm_type == "layer":
            norm_layer = nn.LayerNorm
        elif norm_type == "rms":
            norm_layer = RMSNorm
        elif norm_type == "dyt":
            norm_layer = DyT
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        self.norm1 = norm_layer(hidden_size)
        self.attn = Attention2D_ROPE(hidden_size, num_heads=num_heads, input_size=input_size, qkv_bias=True,
                                     qk_norm=qk_norm)
        self.norm2 = norm_layer(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size,mlp_hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.attn.reset_parameters()

    def _forward(self, x, freq_cis_img=True):
        assert x.ndim == 4, f"x shape must be (b c h w)"
        b, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.attn(self.norm1(x), freq_cis_img)[0]
        x = x + self.mlp(self.norm2(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

    def forward(self, x, freq_cis_img=True):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,  freq_cis_img)
        return self._forward(x,freq_cis_img)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm = DyT_4D(dim)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.conv1)
        apply_initialization(self.conv2)

    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = out + skip  # Residual connection
        return out


class DownBlock(nn.Module):
    def __init__(self, dim):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.residual_block = ResidualBlock(dim)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.conv)
        self.residual_block.reset_parameters()

    def forward(self, x):
        out = self.conv(x)
        out = self.residual_block(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, dim):
        super(UpBlock, self).__init__()
        self.transconv = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.residual_block = ResidualBlock(dim)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.transconv)
        self.residual_block.reset_parameters()

    def forward(self, x):
        out = self.transconv(x)
        out = self.residual_block(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.fc1)
        apply_initialization(self.fc2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, d_model, ratio=4., attn_shortcut=True):
        super(ChannelAttention, self).__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(d_model, int(d_model // ratio), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(d_model // ratio), d_model, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        x = self.sigmoid(out) * x
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class SpatialAttention(nn.Module):
    def __init__(self, c_in, c_hid, c_out, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.proj_1 = nn.Conv2d(c_in, c_hid, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.proj_2 = nn.Conv2d(c_hid, c_out, 1)  # 1x1 conv

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj_1(x)
        x = self.activation(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv1(att)
        x = self.sigmoid(att) * x
        x = self.proj_2(x)
        return x


class Dilation_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        if kernel_size == 3:
            dd_k = kernel_size
            dilation = 1
        else:
            dd_k = (kernel_size - 1) * dilation + 1
        dd_p = dd_k // 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=dd_p,
                               groups=out_channels,
                               dilation=dilation)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.conv1)
        apply_initialization(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LocalFeatureBlock(nn.Module):

    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 9], mlp_ratio=4.,
                 dilation=2,
                 act_layer=nn.GELU, drop=0., drop_path=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1)
        layers = []
        for ker in incep_ker:
            layers.append(nn.Sequential(Dilation_Conv2d(C_hid, C_hid, ker, dilation=dilation),
                                        ChannelAttention(C_hid, ratio=mlp_ratio)
                                        ))
        self.layers = nn.ModuleList(layers)
        self.SpatialAttention = SpatialAttention(len(incep_ker) * C_hid, C_hid, C_out)
        self.norm = DyT_4D(C_out)
        mlp_hidden_dim = int(C_hid * mlp_ratio)
        self.mlp = Mlp(C_out, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def process_first(self, x):
        y = []
        x = self.conv1(x)
        for layer in self.layers:
            y.append(layer(x))
        x = torch.cat(y, dim=1)
        x = self.SpatialAttention(x)
        return x

    def forward(self, x):
        x = self.drop_path(self.process_first(x))
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


class AFNO2DLayer(nn.Module):
    def __init__(self, dim, fno_blocks=8, fno_softshrink=0.01, ):
        super().__init__()
        self.hidden_size = dim

        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.softshrink = fno_softshrink
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.w1, 0)
        nn.init.constant_(self.w2, 0)
        nn.init.constant_(self.b1, 0)
        nn.init.constant_(self.b2, 0)

    @staticmethod
    def multiply(input_data, weights):
        return torch.einsum('...bd,bdk->...bk', input_data, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        bias = x

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x.float()
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0]

        x_imag = self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1]

        x_real = F.relu(x_real)
        x_imag = F.relu(x_imag)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]
        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)
        return x + bias


class FFTBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_blocks: int = 8,
            mlp_ratio: float = 4.0,
            drop: float = 0.1,
            activation_fn=nn.GELU,
            norm_layer=DyT_4D,
            double_skip: bool = True,
            sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)

        self.filter = AFNO2DLayer(embed_dim, num_blocks, sparsity_threshold)
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_latent_dim,
            out_features=embed_dim,
            act_layer=activation_fn,
            drop=drop
        )

        self.double_skip = double_skip
        self.reset_parameters()

    def reset_parameters(self):
        self.filter.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class GaFFU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.local_block = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )
        self.global_block = nn.AdaptiveAvgPool2d(1)

        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.conv1)
        apply_initialization(self.conv2)
        apply_initialization(self.local_block[0])
        apply_initialization(self.local_block[1])

    def forward(self, global_feature, local_feature):
        x = torch.cat((global_feature, local_feature), dim=1)
        x1 = self.conv1(x)
        x1 = self.sigmoid(x1)
        R_g, R_l = torch.chunk(x1, 2, dim=1)  # 重置门
        x = torch.cat([R_g * global_feature, R_l * local_feature], dim=1)
        x = self.conv2(x)
        x_g = self.global_block(x)
        x_l = self.local_block(x)
        w = self.sigmoid(x_g + x_l)
        x = w * x + (1 - w) * (global_feature + local_feature)
        return x


class SingleModal_block_no_t(nn.Module):
    def __init__(self, dim, mlp_ratio=4., num_blocks=8,
                 sparsity_threshold=0.01,
                 double_skip=True,
                 incep_ker=[3, 5, 7, 9],
                 drop=0.1, drop_path=0.1,
                 dilation=2,
                 act_layer=nn.GELU,
                 norm_layer=DyT_4D, **kwargs):
        super().__init__()
        self.global_fusion = FFTBlock(embed_dim=dim,
                                      num_blocks=num_blocks,
                                      mlp_ratio=mlp_ratio,
                                      drop=drop,
                                      activation_fn=act_layer,
                                      norm_layer=norm_layer,
                                      double_skip=double_skip,
                                      sparsity_threshold=sparsity_threshold
                                      )
        self.local_fusion = LocalFeatureBlock(dim,
                                              dim,
                                              dim,
                                              incep_ker=incep_ker,
                                              mlp_ratio=4.,
                                              act_layer=act_layer,
                                              dilation=dilation,
                                              drop=drop, drop_path=drop_path)
        self.gaFFU = GaFFU(dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.global_fusion.reset_parameters()
        self.local_fusion.reset_parameters()
        self.gaFFU.reset_parameters()

    def forward(self, x):
        x1 = self.global_fusion(x)
        x2 = self.local_fusion(x)
        x = self.gaFFU(x1, x2)
        return x


class Basic_block(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid,
                 num_heads=8,
                 input_size=(128, 128),
                 depth=[1, 2, 4, 2, 1],
                 channel_out=None,
                 mlp_ratio=4.,
                 drop=0.0,
                 drop_path=0.1,
                 num_blocks=8,
                 sparsity_threshold=0.01,
                 double_skip=True,
                 incep_ker=[3, 5, 7, 11],
                 use_checkpoint=False,
                 norm_type="dyt",
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if channel_out is None:
            channel_out = channel_in
        self.conv_in = nn.Conv2d(channel_in, channel_hid, 1)
        self.layer0 = nn.ModuleList([SingleModal_block_no_t(channel_hid,
                                                            mlp_ratio=mlp_ratio,
                                                            num_blocks=num_blocks,
                                                            sparsity_threshold=sparsity_threshold,
                                                            double_skip=double_skip,
                                                            incep_ker=incep_ker,
                                                            drop=drop, drop_path=drop_path,
                                                            act_layer=nn.GELU,
                                                            norm_layer=DyT_4D
                                                            )
                                     for _ in range(depth[0])])
        self.down_0_1 = DownBlock(channel_hid)

        self.layer1 = nn.ModuleList([SingleModal_block_no_t(channel_hid,
                                                            mlp_ratio=mlp_ratio,
                                                            num_blocks=num_blocks,
                                                            sparsity_threshold=sparsity_threshold,
                                                            double_skip=double_skip,
                                                            incep_ker=incep_ker,
                                                            drop=drop, drop_path=drop_path,
                                                            act_layer=nn.GELU,
                                                            norm_layer=DyT_4D
                                                            )
                                     for _ in range(depth[1])])
        self.down_1_2 = DownBlock(channel_hid)
        self.reduce_1_2 = nn.Conv2d(channel_hid, channel_hid * 2, kernel_size=1)

        self.latent = nn.ModuleList([
            SpaceTransformerBlock(hidden_size=channel_hid * 2,
                                  num_heads=num_heads,
                                  input_size=(input_size[0] // (2 ** 2), input_size[1] // (2 ** 2)),
                                  mlp_ratio=mlp_ratio,
                                  qk_norm=False,
                                  norm_type=norm_type,
                                  use_checkpoint=use_checkpoint
                                  ) for _ in
            range(depth[2])
        ])

        self.up_2_3 = nn.Sequential(
            nn.Conv2d(channel_hid * 2, channel_hid, kernel_size=1),
            UpBlock(channel_hid)
        )
        self.reduce_2_3 = nn.Sequential(
            DyT_4D(channel_hid * 2),
            nn.Conv2d(channel_hid * 2, channel_hid, kernel_size=1),
            nn.GELU()
        )
        self.layer3 = nn.ModuleList([SingleModal_block_no_t(channel_hid,
                                                            mlp_ratio=mlp_ratio,
                                                            num_blocks=num_blocks,
                                                            sparsity_threshold=sparsity_threshold,
                                                            double_skip=double_skip,
                                                            incep_ker=incep_ker,
                                                            drop=drop, drop_path=drop_path,
                                                            act_layer=nn.GELU,
                                                            norm_layer=DyT_4D
                                                            )
                                     for _ in range(depth[3])])
        self.up_3_4 = UpBlock(channel_hid)
        self.reduce_3_4 = nn.Sequential(
            DyT_4D(channel_hid * 2),
            nn.Conv2d(channel_hid * 2, channel_hid, kernel_size=1),
            nn.GELU()
        )
        self.layer4 = nn.ModuleList([SingleModal_block_no_t(channel_hid,
                                                            mlp_ratio=mlp_ratio,
                                                            num_blocks=num_blocks,
                                                            sparsity_threshold=sparsity_threshold,
                                                            double_skip=double_skip,
                                                            incep_ker=incep_ker,
                                                            drop=drop, drop_path=drop_path,
                                                            act_layer=nn.GELU,
                                                            norm_layer=DyT_4D
                                                            )
                                     for _ in range(depth[4])])

        self.conv_out = nn.Conv2d(channel_hid, channel_out, 1)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.conv_in)
        apply_initialization(self.conv_out)
        layers = self.layer0 + self.layer1 + self.latent + self.layer3 + self.layer4
        for m in layers:
            m.reset_parameters()

    def process_block(self, x, blocks):
        for block in blocks:
            x = block(x)
        return x

    def _forward(self, x):
        x = self.conv_in(x)
        input0 = self.process_block(x, self.layer0)
        input1 = self.down_0_1(input0)
        input1 = self.process_block(input1, self.layer1)
        input2 = self.down_1_2(input1)
        input2 = self.reduce_1_2(input2)
        input2 = self.process_block(input2, self.latent)
        output3 = self.up_2_3(input2)
        output3 = torch.cat([output3, input1], dim=1)
        output3 = self.reduce_2_3(output3)
        output3 = self.process_block(output3, self.layer3)
        output4 = self.up_3_4(output3)
        output4 = torch.cat([output4, input0], dim=1)
        output4 = self.reduce_3_4(output4)
        output4 = self.process_block(output4, self.layer4)
        x = self.conv_out(output4)
        return x

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)


if __name__ == '__main__':
    model = Basic_block(255, 192, channel_out=192).cuda()
    y = torch.rand(1, 255, 128, 128).cuda()
    with torch.inference_mode():
        print(model(y).shape)
