import torch.nn as nn
import torch
from einops import repeat, rearrange
import numpy as np
from .rope import apply_rotary_emb, get_1d_rotary_pos_embed, get_2d_rotary_pos_embed_lumina
from .utils import apply_initialization


def get_attn_subsequence_mask(seq, head_first=True):
    '''
    seq shape is (b h l c)
    '''
    assert seq.ndim == 4, 'seq shape must be (b h l c) or (b l h c)'
    if not head_first:
        seq = seq.transpose(1, 2)
    b, h, l, _ = seq.shape
    attn_shape = [b, l, l]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).bool()
    subsequence_mask = subsequence_mask.to(seq.device)
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class CrossAttention(nn.Module):
    """
    Use QK Normalization.
    """

    def __init__(self,
                 qdim,
                 kdim,
                 num_heads,
                 qkv_bias=True,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer=nn.LayerNorm,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, freqs_cis_img=True):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]
        _, s2, c = y.shape  # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(y).view(b, s2, 2, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        k, v = kv.unbind(dim=1)  # [b, h,s d]
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cis_img:
            freqs_cis = get_1d_rotary_pos_embed(self.head_dim, s1).to(x.device)
            qq = apply_rotary_emb(q, freqs_cis, use_real=False)
            assert qq.shape == q.shape, f'qq: {qq.shape}, q: {q.shape}'
            q = qq
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [b, h, s, d] @ [b, h, d, l]
        attn = attn.softmax(dim=-1)  # [b, h, s, l]
        attn = self.attn_drop(attn)
        x = attn @ v  # [b, h, s, d]

        x = x.transpose(1, 2).reshape(b, s1, c)
        x = self.out_proj(x)
        x = self.proj_drop(x)

        out_tuple = (x,)

        return out_tuple


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention

        self.scale = self.head_dim ** -0.5

        # qkv --> Wqkv
        self.Wqkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis_img=True, mask_attention=False):
        B, N, C = x.shape
        qkv = self.Wqkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, b, h, s, d]
        q, k, v = qkv.unbind(0)  # [b, h, s, d]
        q = self.q_norm(q)  # [b, h, s, d]
        k = self.k_norm(k)  # [b, h, s, d]

        # Apply RoPE if needed,head_first must be true
        if freqs_cis_img:
            freqs_cis = get_1d_rotary_pos_embed(self.head_dim, N).to(x.device)
            qq = apply_rotary_emb(q, freqs_cis, use_real=False)
            kk = apply_rotary_emb(k, freqs_cis, use_real=False)
            assert qq.shape == q.shape and kk.shape == k.shape, \
                f'qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}'
            q, k = qq, kk

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [b, h, s, d] @ [b, h, d, s]
        if mask_attention:
            mask = get_attn_subsequence_mask(q, head_first=True)
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, -1e9)
        attn = attn.softmax(dim=-1)  # [b, h, s, s]

        attn = self.attn_drop(attn)
        x = attn @ v  # [b, h, s, d]

        x = x.transpose(1, 2).reshape(B, N, C)  # [b, s, h, d]
        x = self.out_proj(x)
        x = self.proj_drop(x)

        out_tuple = (x,)

        return out_tuple


class Attention2D_ROPE(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(self, dim, num_heads, input_size, qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.h, self.w = input_size
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention

        self.scale = self.head_dim ** -0.5

        # qkv --> Wqkv
        self.Wqkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.Wqkv)
        apply_initialization(self.out_proj)

    def forward(self, x, freqs_cis_img=True):
        B, N, C = x.shape
        assert N == self.h * self.w, 'wrong size for x'
        qkv = self.Wqkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, b, h, s, d]
        q, k, v = qkv.unbind(0)  # [b, h, s, d]
        q = self.q_norm(q)  # [b, h, s, d]
        k = self.k_norm(k)  # [b, h, s, d]

        # Apply RoPE if needed,head_first must be true
        if freqs_cis_img:
            q, k = map(lambda x: rearrange(x, 'b h (H W) d -> b h H W d', H=self.h, W=self.w), [q, k])
            freqs_cis = get_2d_rotary_pos_embed_lumina(self.head_dim, self.h, self.w).to(x.device)
            qq = apply_rotary_emb(q, freqs_cis, use_real=False)
            kk = apply_rotary_emb(k, freqs_cis, use_real=False)
            assert qq.shape == q.shape and kk.shape == k.shape, \
                f'qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}'
            q, k = qq, kk
        q, k = map(lambda x: rearrange(x, 'b h H W d -> b h (H W) d'), [q, k])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [b, h, s, d] @ [b, h, d, s]
        attn = attn.softmax(dim=-1)  # [b, h, s, s]
        attn = self.attn_drop(attn)
        x = attn @ v  # [b, h, s, d]

        x = x.transpose(1, 2).reshape(B, N, C)  # [b, s, h, d]
        x = self.out_proj(x)
        x = self.proj_drop(x)

        out_tuple = (x,)

        return out_tuple


class CrossAttention2D_ROPE(nn.Module):
    """
    Use QK Normalization.
    """

    def __init__(self,
                 qdim,
                 kdim,
                 num_heads,
                 input_size,
                 qkv_bias=True,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer=nn.LayerNorm,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.h, self.w = input_size
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, freqs_cis_img=True):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]
        _, s2, c = y.shape  # [b, s2, 1024]
        assert s1 == self.h * self.w, 'wrong size for x'
        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(y).view(b, s2, 2, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        k, v = kv.unbind(dim=1)  # [b, h,s d]
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cis_img:
            q = rearrange(q, 'b h (H W) d -> b h H W d', H=self.h, W=self.w)
            freqs_cis = get_2d_rotary_pos_embed_lumina(self.head_dim, self.h, self.w).to(x.device)
            qq = apply_rotary_emb(q, freqs_cis, use_real=False)
            assert qq.shape == q.shape, f'qq: {qq.shape}, q: {q.shape}'
            q = qq
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [b, h, s, d] @ [b, h, d, l]
        attn = attn.softmax(dim=-1)  # [b, h, s, l]
        attn = self.attn_drop(attn)
        x = attn @ v  # [b, h, s, d]

        x = x.transpose(1, 2).reshape(b, s1, c)
        x = self.out_proj(x)
        x = self.proj_drop(x)

        out_tuple = (x,)

        return out_tuple
