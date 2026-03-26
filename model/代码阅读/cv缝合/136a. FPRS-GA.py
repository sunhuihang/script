# 136b_FPRS_GA.py
# 哔哩哔哩/微信公众号: CV缝合救星独家复现
# FPRS-GA: Frequency-guided Pixel Refinement with Spectral Gated Aggregation
# 频引导像素精炼 + 频谱门控聚合（CVPR风格模块）

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPRS_GA(nn.Module):
    """
    FPRS-GA（Frequency-guided Pixel Refinement with Spectral Gated Aggregation）
    设计动机（人话版）：
    1）空间域：深度可分卷积 + 1x1 卷积，低成本抓局部纹理，这一支路等价于SPR-SA的“像素精炼”主干。
    2）频域：FFT把特征映射到频谱，把实部/虚部当成2个通道堆起来，用1x1卷积“调制频谱”，再IFFT回空间。
    3）门控聚合：引入可学习系数 gamma/beta，频域与空间域动态平衡，避免频域过强带来的伪影回灌。
    4）通道注意力：GAP + Softmax 做通道权重，对应论文里“每个像素在同一位置感知不同退化信号”的直觉。
    """
    def __init__(self, dim: int, growth_rate: float = 2.0):
        super().__init__()
        # 隐藏通道（和原SPR-SA保持一致策略）
        hidden_dim = int(dim * growth_rate)

        # ===== 空间域：局部像素精炼（CV缝合救星保留件） =====
        # 深度可分卷积提局部、1x1卷积做通道混合
        self.local_refine = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=dim),  # DWConv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)       # PWConv
        )

        # ===== 频域：实/虚拼接 → 频谱调制 → 复原复数谱 → IFFT =====
        # 注意：在频域我们不做复数卷积，而是把 real/imag 当作 2*hidden_dim 个实通道来做线性变换
        self.freq_proj = nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=1, stride=1, padding=0)

        # ===== 通道注意力（GAP + Softmax），就是CV缝合救星的“通道加权法宝” =====
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0)

        # 频域/空间门控系数（可学习），初始化小一点更稳（频域先浅尝辄止）
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 频域分支权重
        self.beta  = nn.Parameter(torch.tensor(1.0))  # 空间分支权重

    @torch.no_grad()
    def _debug_shapes(self, name, x):
        # 可选：调试时打开，打印形状
        # print(f"[debug] {name}: {tuple(x.shape)}")
        pass

    def _fft_modulate(self, x_local: torch.Tensor) -> torch.Tensor:
        """
        频域路径：
        1) FFT 得到复数谱 X（B,C,H,W），分解为 real/imag 两个实张量
        2) 在 [real, imag] 拼接的 2C 通道上做 1x1 卷积进行频谱“调制”
        3) 把调制后的实/虚再组成复数谱 X'，IFFT 回空间，取 real 作为增强特征
        """
        # X: 复数谱（complex64/complex32）
        X = torch.fft.fft2(x_local, norm='ortho')
        real = X.real
        imag = X.imag
        self._debug_shapes("fft_real", real)
        self._debug_shapes("fft_imag", imag)

        # 通道拼接 [B, 2C, H, W]
        freq_cat = torch.cat([real, imag], dim=1)

        # 1x1 卷积做频谱调制（相当于对实/虚共同做线性组合）
        freq_mod = self.freq_proj(freq_cat)

        # 切回实/虚
        c = freq_mod.shape[1] // 2
        real_mod, imag_mod = freq_mod[:, :c], freq_mod[:, c:]

        # 复原复数谱，IFFT回空间
        X_mod = torch.complex(real_mod, imag_mod)
        x_freq = torch.fft.ifft2(X_mod, norm='ortho').real  # 只取实部，数值更稳
        return x_freq

    def _channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        通道注意力（GAP + Softmax），让每个通道学会“自我权重”
        这一步就是经典的“CV缝合救星”通道重加权——既简单、又好使。
        """
        w = F.adaptive_avg_pool2d(x, (1, 1))           # [B,C,1,1]
        w = F.softmax(w, dim=1)                        # 通道维归一化
        return x * w                                   # 按通道乘权

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：x [B, dim, H, W]
        输出：y [B, dim, H, W]
        """
        # 1) 空间局部精炼（低成本、强鲁棒）：DWConv + PWConv
        x_local = self.local_refine(x)                 # [B, hidden, H, W]
        self._debug_shapes("x_local", x_local)

        # 2) 频域增强：FFT → 频谱调制 → IFFT（只取real）
        x_freq = self._fft_modulate(x_local)           # [B, hidden, H, W]
        self._debug_shapes("x_freq", x_freq)

        # 3) 通道注意力（CV缝合救星的拿手戏）
        x_local = self._channel_attention(x_local)     # [B, hidden, H, W]

        # 4) 频-空门控聚合（Spectral Gated Aggregation）
        fused = self.gamma * x_freq + self.beta * x_local

        # 5) 激活 + 映射回输入通道数
        fused = self.act(fused)
        y = self.out_proj(fused)                       # [B, dim, H, W]
        return y


# ===================== 可直接运行的自测脚本 =====================
if __name__ == "__main__":
    torch.manual_seed(0)

    # 模拟一份输入：B=1, C=64, H=W=32
    x = torch.randn(1, 64, 32, 32)

    # 创建模型（和你之前的SPR-SA保持接口一致：dim/growth_rate）
    model = FPRS_GA(dim=64, growth_rate=2.0)

    # 打印结构
    print("FPRS-GA（Frequency-guided Pixel Refinement with Spectral Gated Aggregation）模型结构：")
    print(model)

    # 前向一次
    y = model(x)

    # 打印输入输出尺寸
    print("\n输入形状：", tuple(x.shape))
    print("输出形状：", tuple(y.shape))

    # 额外打印一下门控参数，看看默认融合权重
    print("\n门控系数：gamma(频域) = {:.3f}, beta(空间) = {:.3f}".format(model.gamma.item(), model.beta.item()))

    print("\n哔哩哔哩/微信公众号: CV缝合救星独家复现")
