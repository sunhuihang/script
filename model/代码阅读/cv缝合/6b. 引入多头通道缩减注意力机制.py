import torch
import torch.nn as nn

"""
创新点 2：多头通道缩减注意力机制
动机：单一头的通道缩减注意力在捕捉全局特征时视角单一，限制了模型对图像多样性特征的理解能力。
在实际的语义分割和目标检测任务中，图像通常包含多种类型的细节特征（如纹理、边缘、颜色等），
单一的缩减方案难以充分表达这种多样性。通过将通道划分为多个子头，并分别进行注意力计算，
可以帮助模型在不同特征上获取多维度的上下文信息，提升在复杂场景中的表现力。

方法：在CRA模块中引入多头机制，将通道划分为若干个子集（即多个“头”），每个子集独立地进行通道
缩减和注意力计算。具体而言，每个头都包含独立的查询、键和值的投影操作，将各自的通道缩减至单维，
随后再在每个头中计算注意力。最终将各个头的输出特征拼接，恢复成完整的通道维度。这样的设计可以
在不同的头中分别捕捉到不同维度的上下文信息，使模型能够在多个尺度和特征类型上进行注意力聚合。

效果：多头通道缩减注意力机制使模型具备了更灵活的表达能力，在保持高效计算的同时，能够适应图像
中丰富的多样化信息。
"""
class MultiHeadCRA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, num_heads=4):
        super(MultiHeadCRA, self).__init__()
        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        reduced_dim = self.head_dim // reduction_ratio

        # 为每个头定义缩减后的查询、键和值投影
        self.query_projections = nn.ModuleList([nn.Linear(self.head_dim, reduced_dim) for _ in range(num_heads)])
        self.key_projections = nn.ModuleList([nn.Linear(self.head_dim, reduced_dim) for _ in range(num_heads)])
        self.value_projections = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        input_flat = x.view(batch_size, channels, -1)

        # 将通道划分为多个头
        head_outputs = []
        for i in range(self.num_heads):
            head_input = input_flat[:, i * self.head_dim: (i + 1) * self.head_dim, :]

            # 获取每个头的查询、键和值
            avg_pool = torch.mean(head_input, dim=-1, keepdim=True)
            query = self.query_projections[i](head_input.permute(0, 2, 1))
            key = self.key_projections[i](avg_pool.permute(0, 2, 1))
            value = self.value_projections[i](avg_pool.permute(0, 2, 1))

            # 计算注意力并应用
            attention_map = torch.softmax(torch.bmm(query, key.permute(0, 2, 1)), dim=1)
            head_output = torch.bmm(attention_map, value).permute(0, 2, 1).contiguous()

            head_outputs.append(head_output)

        # 合并所有头并还原形状
        out = torch.cat(head_outputs, dim=1)
        out = out.view(batch_size, channels, height, width)
        return out
if __name__ == "__main__":
    input = torch.randn(8, 64, 32, 32)
    CRA = MultiHeadCRA(in_channels=64, reduction_ratio=4)
    output = CRA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())