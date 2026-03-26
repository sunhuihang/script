import math
import torch
import torch.nn as nn
'''
ESSA: Efficient Self-Attention Module for Hyperspectral Image Super-Resolution (ICCV 2023)
即插即用模块：ESSA（替身模块）
一、背景
1. CNN 局限催生变革契机：传统基于 CNN 的单高光谱图像超分辨率（single-HSI-SR）方法在构建长程依赖和捕捉光谱
特征交互信息方面存在缺陷。其卷积核聚焦局部特征，致使网络感受野受限，无法充分利用光谱信息，超分辨率后易出现
伪影，难以满足实际需求。
2. ESSA 应对挑战而生：鉴于 Vision Transformers 在处理长程依赖方面的优势及在 HSI-SR 应用中的困境，如数据需
求大、自注意力计算复杂等，ESSA 模块被提出。旨在为 HSI-SR 任务量身打造高效注意力机制，提升模型性能。

二、模块原理
1. 基于光谱特性的 SCC 度量引入
a. 光谱友好相似性度量：利用光谱相关系数（SCC）替代传统注意力矩阵。SCC 基于皮尔逊相关系数，考量光谱曲线特性，对
因遮挡或阴影引发的光谱幅度变化不敏感，引入通道方向的归纳偏差，有效提升模型训练效率，使其能在小数据集上从头训练。
b. 归纳偏差优势体现：SCC 具有通道平移不变性，即对光谱数据的缩放和平移变换保持稳定，使得模型在复杂光照条件下的 
HSI 数据中能聚焦关键特征，增强模型鲁棒性与泛化能力。
2. 高效 SCC - 核注意力机制构建
a. 核技巧降维计算：为缓解计算负担，将 SCC 融入非线性平方指数核（Mercer 核）。依据 Mercer 定理，将 SCC 表示为
可通过泰勒展开获取的映射函数的点积形式，重新规划计算流程，改变自注意力乘法顺序，使计算复杂度从二次降至线性，显著
减少计算量。
b. 理论支撑保障实现：通过严格的数学推导，证明 SCC 核满足 Mercer 核条件，确保映射函数存在。在实际计算中，合理选
择泰勒展开阶数以平衡性能与计算成本，实现高效计算与准确建模的统一。

三、适用于：单高光谱图像超分辨率（single-HSI-SR）任务，在该任务中作为核心注意力模块提升模型对空间 - 光谱信息的利
用效率，优化超分辨率重建效果。



'''
class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b,c,h,w=x.shape
        x = x.reshape(b,c,h*w).permute(0,2,1)
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = t1 + t2
        attn = self.ln(attn)
        x = attn.reshape(b,h,w,c).permute(0,3,1,2)
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    input =  torch.randn(1, 32, 64, 64)
    model = ESSAttn(32)
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())
