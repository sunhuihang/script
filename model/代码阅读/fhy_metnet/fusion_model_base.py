import math

import torch
from model.attn_layers import Attention2D_ROPE
from model.norm_layers import RMSNorm, ChanLayerNorm, DyT, DyT_4D
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import numpy as np
from einops.layers.torch import Rearrange

try:
    from .gate_blocks import Basic_block, SpatialAttention, Dilation_Conv2d, ChannelAttention, SwiGLUFFN, GaFFU
    from .layers import *

except:
    from gate_blocks import Basic_block, SpatialAttention, Dilation_Conv2d, ChannelAttention, SwiGLUFFN, GaFFU
    from layers import *


class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).to(origin_dtype)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True, spec_norm=False, down=True):
        if down:
            samplings = sampling_generator(N_S)
        else:
            samplings = [False] * N_S
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                   act_inplace=act_inplace, spec_norm=spec_norm),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace, spec_norm=spec_norm) for s in samplings[1:]]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.enc:
            m.reset_parameters()

    def forward(self, x):
        for i in range(len(self.enc)):
            x = self.enc[i](x)
        return x


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True, skip_channel=None, skip=False):
        if skip_channel is None:
            skip_channel = C_hid
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.skip = skip

        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
            ConvSC(C_hid + skip_channel if skip else C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                   act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dec:
            m.reset_parameters()
            # apply_initialization(self.readout,)
        nn.init.constant_(self.readout.weight, 0.05)
        nn.init.constant_(self.readout.bias, 0)

    def forward(self, hid, rad=None):
        for i in range(len(self.dec)):
            if i == len(self.dec) - 1 and self.skip:
                hid = self.dec[i](torch.cat((hid, rad), dim=1))
            else:
                hid = self.dec[i](hid)
        Y = self.readout(hid)
        return Y


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channel):
        super().__init__()
        self.norm_final = DyT(final_hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )
        self.patch_size = patch_size
        self.linear = nn.Linear(final_hidden_size, out_channel * patch_size ** 2)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.linear)

    def forward(self, x, c):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        x = rearrange(x, 'b (h w) (c p q) -> b c (h p) (w q)',
                      h=h,
                      w=w,
                      p=self.patch_size,
                      q=self.patch_size)
        return x


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
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.residual_block = ResidualBlock(dim * 2)
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
        self.transconv = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.residual_block = ResidualBlock(dim // 2)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.transconv)
        self.residual_block.reset_parameters()

    def forward(self, x):
        out = self.transconv(x)
        out = self.residual_block(out)
        return out


class GlobalBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            mod_dim: int,
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

        self.filter = ModAFNO2DLayer(embed_dim, mod_dim, num_blocks, sparsity_threshold)
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        self.mlp = ModAFNOMlp(
            in_features=embed_dim,
            hidden_features=mlp_latent_dim,
            out_features=embed_dim,
            mod_features=mod_dim,
            act_layer=activation_fn,
            drop=drop
        )

        self.double_skip = double_skip
        self.reset_parameters()

    def reset_parameters(self):
        self.filter.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, mod_embed):
        residual = x
        x = self.norm1(x)
        x = self.filter(x, mod_embed)

        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.mlp(x, mod_embed)
        x = x + residual
        return x


class LocalBlock(nn.Module):

    def __init__(self, C_in, C_hid, C_out, tim_dim, incep_ker=[3, 5, 7, 11], mlp_ratio=4.,
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
        self.mlp = ModAFNOMlp(C_out, mlp_hidden_dim, mod_features=tim_dim, act_layer=act_layer, drop=drop)
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

    def forward(self, x, t):
        x = self.drop_path(self.process_first(x))
        x = x + self.drop_path(self.mlp(self.norm(x), t))
        return x


class SingleModal_block(nn.Module):
    def __init__(self, dim, mod_dim, mlp_ratio=4., num_blocks=8,
                 sparsity_threshold=0.01,
                 double_skip=True,
                 incep_ker=[3, 5, 7, 11],
                 drop=0.1, drop_path=0.1,
                 act_layer=nn.GELU,
                 dilation=2,
                 norm_layer=DyT_4D, **kwargs):
        super().__init__()
        self.global_fusion = GlobalBlock(embed_dim=dim,
                                         mod_dim=mod_dim,
                                         num_blocks=num_blocks,
                                         mlp_ratio=mlp_ratio,
                                         drop=drop,
                                         activation_fn=act_layer,
                                         norm_layer=norm_layer,
                                         double_skip=double_skip,
                                         sparsity_threshold=sparsity_threshold
                                         )
        self.local_fusion = LocalBlock(dim,
                                       dim,
                                       dim,
                                       tim_dim=mod_dim,
                                       incep_ker=incep_ker,
                                       mlp_ratio=4.,
                                       act_layer=act_layer,
                                       dilation=dilation,
                                       drop=drop, drop_path=drop_path)
        self.norm = norm_layer(dim)
        self.gaFFU = GaFFU(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ModAFNOMlp(in_features=dim, hidden_features=mlp_hidden_dim, mod_features=mod_dim,
                              out_features=dim, act_layer=act_layer,
                              drop=drop)
        self.reset_parameters()

    def reset_parameters(self):
        self.global_fusion.reset_parameters()
        self.local_fusion.reset_parameters()
        self.gaFFU.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, t):
        x1 = self.global_fusion(x, t)
        x2 = self.local_fusion(x, t)
        x = self.gaFFU(x1, x2)
        x = x + self.mlp(self.norm(x), t)
        return x


class SpaceTransformerBlock(nn.Module):
    """
       A DiT block with `add` conditioning.
       """

    def __init__(self,
                 hidden_size,
                 c_emb_size,
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
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim)
        # Simply use add like SDXL.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, hidden_size * 6, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        self.attn.reset_parameters()

    def _forward(self, x, c=None, freq_cis_img=True):
        assert x.ndim == 4, f"x shape must be (b c h w)"
        b, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * \
            self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freq_cis_img)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

    def forward(self, x, c=None, freq_cis_img=True):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, c, freq_cis_img)
        return self._forward(x, c, freq_cis_img)


class Modal_block(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2, mod_dim, channel_out=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1,
                 num_blocks=8,
                 sparsity_threshold=0.01,
                 double_skip=True,
                 dilation=2,
                 incep_ker=[3, 5, 7, 11],
                 use_checkpoint=False,
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if channel_out is None:
            channel_out = channel_in
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]
        self.conv_in = nn.Conv2d(channel_in, channel_hid, 1)
        self.norm = DyT_4D(channel_in)
        self.act = nn.GELU()
        self.layers = nn.ModuleList([SingleModal_block(channel_hid, mod_dim,
                                                       mlp_ratio=mlp_ratio,
                                                       num_blocks=num_blocks,
                                                       sparsity_threshold=sparsity_threshold,
                                                       double_skip=double_skip,
                                                       incep_ker=incep_ker,
                                                       drop=drop, drop_path=dpr[i],
                                                       act_layer=nn.GELU,
                                                       dilation=dilation,
                                                       norm_layer=DyT_4D
                                                       )
                                     for i in range(N2)])
        self.conv_out = nn.Conv2d(channel_hid, channel_out, 1)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.conv_in)
        apply_initialization(self.conv_out)
        for m in self.layers:
            m.reset_parameters()

    def _forward(self, x, t):
        for i in range(self.N2):
            if i == 0:
                x = self.act(self.conv_in(self.norm(x)))
            x = self.layers[i](x, t)
        x = self.conv_out(x)
        return x

    def forward(self, x, t):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, t)
        return self._forward(x, t)


class Force_Mlp(nn.Module):
    def __init__(self, in_features, mlp_radio=4., out_features=None,
                 bottleneck_dim=256,
                 act_layer=nn.GELU, target_size=(37, 37)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_radio)
        self.target_size = target_size
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, 1)
        self.act2 = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, bottleneck_dim, 1)
        self.last_layer = nn.Conv2d(bottleneck_dim, out_features, 1, bias=False)

    def forward(self, x):
        x = F.interpolate(x.float(), size=self.target_size, mode='bilinear')
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


class FusionModel_v4(nn.Module):
    def __init__(self, dbz_shape, era5_shape, geo_c,
                 hid_S=16, N_S=4, drop=0.1, drop_path=0.2,
                 mlp_ratio=4., spatio_kernel_enc=3,
                 input_size=(512, 512),
                 num_blocks=8,
                 incep_ker=[3, 5, 7, 11],
                 sparsity_threshold=0.01,
                 double_skip=True,
                 init_depth=[1, 2, 4, 2, 1],
                 depth=[1, 2, 4, 2, 1],
                 out_dbz=1,
                 embed_dim=192,
                 num_heads=16,
                 act_inplace=True,
                 norm_type="dyt",
                 use_checkpoint=False
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.hid_S = hid_S
        self.out_dbz = out_dbz
        dbz_t, dbz_c = dbz_shape
        era5_t, era5_c = era5_shape
        self.enc_geo = Encoder(geo_c, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_dbz = Encoder(dbz_c, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_era5 = Encoder(era5_c, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)

        self.t_embedder_1 = TimestepEmbedder(embed_dim)
        self.t_embedder_2 = TimestepEmbedder(embed_dim)
        self.t_embedder_3 = TimestepEmbedder(embed_dim)

        self.init_size = (input_size[0] // (2 ** (N_S // 2)), input_size[1] // (2 ** (N_S // 2)))
        self.mixer_init_field = Basic_block(
            (dbz_t + era5_t + 1) * hid_S, embed_dim,
            num_heads=num_heads,
            input_size=self.init_size,
            depth=init_depth,
            channel_out=embed_dim,
            incep_ker=incep_ker,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            double_skip=double_skip,
            norm_type=norm_type,
            use_checkpoint=use_checkpoint
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.init_size), requires_grad=False)
        self.encoder_level_1 = Modal_block(
            embed_dim, embed_dim, N2=depth[0],
            mod_dim=embed_dim,
            channel_out=embed_dim,
            incep_ker=incep_ker,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            double_skip=double_skip,
            use_checkpoint=use_checkpoint
        )
        self.down1_2 = DownBlock(embed_dim)
        self.encoder_level_2 = Modal_block(
            embed_dim * 2, embed_dim, N2=depth[1],
            mod_dim=embed_dim,
            channel_out=embed_dim,
            incep_ker=incep_ker,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            double_skip=double_skip,
            use_checkpoint=use_checkpoint
        )
        self.down2_3 = DownBlock(embed_dim)
        # latent
        scale_size = 2 ** (N_S // 2 + len(depth) // 2)
        scale_h = input_size[0] // scale_size
        scale_w = input_size[1] // scale_size
        self.latent = nn.ModuleList([
            SpaceTransformerBlock(hidden_size=embed_dim * 2,
                                  c_emb_size=embed_dim,
                                  num_heads=num_heads,
                                  input_size=(scale_h, scale_w),
                                  mlp_ratio=mlp_ratio,
                                  qk_norm=False,
                                  norm_type=norm_type,
                                  use_checkpoint=use_checkpoint
                                  ) for _ in
            range(depth[2])
        ])
        self.force_latent = Force_Mlp(embed_dim * 2, mlp_radio=mlp_ratio, out_features=1024)
        self.up3_2 = UpBlock(embed_dim * 2)  ## From Level 3 to Level 2
        self.decoder_level_2 = Modal_block(
            embed_dim * 2, embed_dim, N2=depth[3],
            mod_dim=embed_dim,
            channel_out=embed_dim * 2,
            incep_ker=incep_ker,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            double_skip=double_skip,
            use_checkpoint=use_checkpoint
        )
        self.up2_1 = UpBlock(embed_dim * 2)  ## From Level 2 to Level 1
        self.decoder_level_1 = Modal_block(
            embed_dim * 2, embed_dim, N2=depth[4],
            mod_dim=embed_dim,
            channel_out=embed_dim,
            incep_ker=incep_ker,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            double_skip=double_skip,
            use_checkpoint=use_checkpoint
        )
        self.t_embedder_4 = TimestepEmbedder(embed_dim)
        self.dec_dbz = FinalLayer(embed_dim, embed_dim, 2 ** (N_S // 2), out_dbz)
        self.reset_parameters()
        # self.freeze_part_weight()

    @staticmethod
    def freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def freeze_part_weight(self):
        # self.freeze(self.t_embedder_1)
        # self.freeze(self.t_embedder_2)
        # self.freeze(self.t_embedder_3)
        #
        # self.freeze(self.enc_geo)
        # self.freeze(self.enc_dbz)
        # self.freeze(self.enc_era5)
        # self.freeze(self.mixer_init_field)
        #
        # self.freeze(self.encoder_level_1)
        # self.freeze(self.down1_2)
        #
        # self.freeze(self.encoder_level_2)
        # self.freeze(self.down2_3)

        # for block in self.latent:
        #     self.freeze(block)
        self.freeze(self.up3_2)
        self.freeze(self.decoder_level_2)
        self.freeze(self.up2_1)
        self.freeze(self.decoder_level_1)
        print('freeze_part_weight is done')

    def reset_parameters(self):
        self.enc_geo.reset_parameters()
        self.enc_dbz.reset_parameters()
        self.enc_era5.reset_parameters()

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[1], self.init_size)
        pos_embed = rearrange(pos_embed, '(h w) c -> c h w', h=self.init_size[0], w=self.init_size[1])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.encoder_level_1.reset_parameters()
        self.down1_2.reset_parameters()

        self.encoder_level_2.reset_parameters()
        self.down2_3.reset_parameters()

        for block in self.latent:
            block.reset_parameters()

        self.up3_2.reset_parameters()
        self.decoder_level_2.reset_parameters()

        self.up2_1.reset_parameters()
        self.decoder_level_1.reset_parameters()

        self.dec_dbz.reset_parameters()

    def multi_scale_fusion(self, fusion, t):
        t1 = self.t_embedder_1(t)  # (N, C, 1, 1)
        t2 = self.t_embedder_2(t)  # (N, C, 1, 1)
        t3 = self.t_embedder_3(t)  # (N, C, 1, 1)
        out_enc_level1 = fusion

        out_enc_level1 = self.encoder_level_1(out_enc_level1, t1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = inp_enc_level2

        out_enc_level2 = self.encoder_level_2(out_enc_level2, t2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = inp_enc_level3
        for block in self.latent:
            latent = block(latent, t3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        out_dec_level2 = inp_dec_level2

        out_dec_level2 = self.decoder_level_2(out_dec_level2, t2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        inp_dec_level1 = self.decoder_level_1(inp_dec_level1, t1)
        out_dec_level1 = inp_dec_level1
        return out_dec_level1, latent

    def get_init_field(self, dbz, ERA5, geo):
        b = dbz.shape[0]
        enc_geo = self.enc_geo(geo)
        enc_dbz = self.enc_dbz(rearrange(dbz, 'b t c h w -> (b t) c h w'))
        enc_era5 = self.enc_era5(rearrange(ERA5, 'b t c h w -> (b t) c h w'))
        enc_dbz = rearrange(enc_dbz, '(b t) c h w -> b (t c) h w', b=b)
        enc_era5 = rearrange(enc_era5, '(b t) c h w -> b (t c) h w', b=b)

        x = torch.cat([enc_dbz, enc_era5, enc_geo], dim=1)

        x = self.mixer_init_field(x)

        return x

    def forward(self, dbz, ERA5, geo, t):
        fusion = self.get_init_field(dbz, ERA5, geo)
        fusion = self.pos_embed + fusion
        out_dec_level1, latent = self.multi_scale_fusion(fusion, t)
        t = self.t_embedder_4(t)
        pred_dbz = self.dec_dbz(out_dec_level1, t)
        latent = self.force_latent(latent)
        return pred_dbz, latent


def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_w)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionModel_v4(dbz_shape=(15, 1), era5_shape=(24, 5), geo_c=3).to(device)
    geo = torch.randn(1, 3, 512, 512).to(device)
    dbz = torch.randn((1, 15, 1, 512, 512)).to(device)
    ERA5 = torch.randn((1, 24, 5, 512, 512)).to(device)
    station = torch.randn((1, 3, 5, 512, 512)).to(device)
    t = torch.randint(0, 120, (1,)).to(device)


    def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()

        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()

        all_size = (param_size + buffer_size) / 1024 / 1024
        return param_sum, buffer_sum, all_size


    print(getModelSize(model)[-1])
    for name, p in model.named_children():
        _, _, a = getModelSize(p)
        print(name, a)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            dbz = model(dbz, ERA5, geo, t)
            print(dbz.shape)
