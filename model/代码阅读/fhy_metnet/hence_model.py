import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class SeparableConv(nn.Module):
    """Dilated 深度可分离卷积：扩大感受野，不降分辨率"""

    def __init__(self, c, dil=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=True)

    def forward(self, x):
        return self.pw(self.conv(x))


class SobelGate(nn.Module):
    """边缘幅度图作空间注意力"""

    def __init__(self, embed_dim, c_embed_dim):
        super().__init__()
        self.weight_x = nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False)
        self.weight_y = nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_embed_dim, embed_dim * 2, bias=True)
        )
        self.att = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(embed_dim // 4, embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gx = F.conv2d(x.mean(dim=1, keepdim=True), self.weight_x, padding=1)
        gy = F.conv2d(x.mean(dim=1, keepdim=True), self.weight_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        gate = self.att(mag)
        return x * gate


class DetailNet(nn.Module):
    def __init__(self, in_c, out_c=None, embed_dim=32):
        super().__init__()
        out_c = out_c or in_c
        self.head = nn.Conv2d(out_c, embed_dim, 3, 1, 1)
        self.body = nn.Sequential(
            *[SeparableConv(embed_dim, dil=2 ** (i % 3) + 1) for i in range(8)]
        )
        self.gate = SobelGate(embed_dim, embed_dim)
        self.tail = nn.Conv2d(embed_dim, out_c, 3, 1, 1)
        # 初始化 tail 为 0 → 训练初期恒等映射
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x, t):
        res = self.tail(self.gate(self.body(self.head(x))))
        return x + res


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dbz = torch.randn((1, 1, 512, 512)).to(device)
    t = torch.randint(0, 120, (1,)).to(device)
    model = DetailNet(1, 1, embed_dim=256).to(device)


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


    print(getModelSize(model))
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            dbz = model(dbz, t)
            print(dbz.shape)
