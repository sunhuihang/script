import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(TemporalAttention, self).__init__()
        # LSTM用于生成时序特征
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, batch_first=True)
        # 用于生成空间注意力的卷积层
        self.conv = nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=3, padding=1)
        # 使用Sigmoid激活函数进行注意力加权
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: 输入的张量，形状为 (batch_size, seq_length, channels, height, width)
        :return: 加权后的输出张量
        """
        batch_size, seq_length, channels, height, width = input_tensor.size()

        # 将输入数据调整为适合LSTM的形状 (batch_size, seq_length, channels*height*width)
        input_flat = input_tensor.view(batch_size, seq_length, -1)  # 每个时间步的特征数是 channels * height * width

        # 使用LSTM提取时序信息
        lstm_out, (hn, cn) = self.lstm(input_flat)  # hn的形状为 (num_layers * num_directions, batch_size, hidden_size)

        # 获取LSTM输出的最后一个时间步的隐藏状态
        time_attention = hn[-1].view(batch_size, 64, 1, 1)  # 假设hidden_size=64, 需要根据具体情况调整

        # 将LSTM的输出映射到空间注意力图
        attention_map = self.conv(time_attention)  # 使用卷积生成空间注意力图，大小为 (batch_size, 1, height, width)
        attention_map = self.sigmoid(attention_map)  # 使用Sigmoid进行归一化

        # 扩展attention_map的维度，使其可以与input_tensor相乘
        attention_map = attention_map.expand(-1, -1, height, width)  # 扩展维度，形状变为 (batch_size, 1, height, width)

        # 将注意力图与输入张量进行逐元素相乘
        output_tensor = input_tensor * attention_map.unsqueeze(1)  # 扩展 attention_map 的维度，以便与 input_tensor 维度匹配

        return output_tensor


# 测试代码
if __name__ == "__main__":
    # 假设输入是一个视频帧序列，大小为 (batch_size, seq_length, channels, height, width)
    input_tensor = torch.randn(8, 10, 64, 32, 32)  # (8, 10, 64, 32, 32)
    print(f"Input shape: {input_tensor.shape}")
    # 创建TemporalAttention模块
    temporal_attention = TemporalAttention(input_channels=64 * 32 * 32, hidden_size=64)

    # 进行前向传播
    output_tensor = temporal_attention(input_tensor)

    print(f"Output shape: {output_tensor.shape}")  # 输出形状应为 (8, 10, 64, 32, 32)
