import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DAGEM(nn.Module):
    def __init__(self, sync_bn=False, input_channels=256):
        super(DAGEM, self).__init__()
        self.input_channels = input_channels
        BatchNorm1d = nn.BatchNorm1d
        BatchNorm2d = nn.BatchNorm2d

        # 边聚合函数
        self.edge_aggregation_func = nn.Sequential(
            nn.Linear(4, 1),
            BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )

        # 节点更新函数
        self.vertex_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.ReLU(inplace=True),
        )

        # 边更新函数
        self.edge_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.ReLU(inplace=True),
        )

        # 边更新缩减函数
        self.update_edge_reduce_func = nn.Sequential(
            nn.Linear(4, 1),
            BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )

        # 用于学习偏移量的卷积层
        self.offset_conv = nn.Conv2d(input_channels, 18, kernel_size=3, padding=1)

        # 可变形卷积层用于增强特征提取
        self.deform_conv = DeformConv2d(input_channels, input_channels, kernel_size=3, padding=1)

        # 最终聚合层
        self.final_aggregation_layer = nn.Sequential(
            nn.Conv2d(input_channels + input_channels // 2, input_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        x = input
        B, C, H, W = x.size()
        vertex = input

        # 构建边特征
        edge = torch.stack(
            (
                torch.cat((input[:, :, -1:], input[:, :, :-1]), dim=2),
                torch.cat((input[:, :, 1:], input[:, :, :1]), dim=2),
                torch.cat((input[:, :, :, -1:], input[:, :, :, :-1]), dim=3),
                torch.cat((input[:, :, :, 1:], input[:, :, :, :1]), dim=3)
            ), dim=-1
        ) * input.unsqueeze(dim=-1)

        # 聚合边特征
        aggregated_edge = self.edge_aggregation_func(
            edge.reshape(-1, 4)
        ).reshape((B, C, H, W))

        # 拼接特征用于节点更新
        cat_feature_for_vertex = torch.cat((vertex, aggregated_edge), dim=1)
        update_vertex = self.vertex_update_func(
            cat_feature_for_vertex.permute(0, 2, 3, 1).reshape((-1, 2 * self.input_channels))
        ).reshape((B, H, W, self.input_channels // 2)).permute(0, 3, 1, 2)

        # 拼接特征用于边更新
        cat_feature_for_edge = torch.cat(
            (
                torch.stack((vertex, vertex, vertex, vertex), dim=-1),
                edge
            ), dim=1
        ).permute(0, 2, 3, 4, 1).reshape((-1, 2 * self.input_channels))
        update_edge = self.edge_update_func(cat_feature_for_edge).reshape((B, H, W, 4, C // 2)).permute(0, 4, 1, 2,
                                                                                                        3).reshape(
            (-1, 4))
        update_edge_converted = self.update_edge_reduce_func(update_edge).reshape((B, C // 2, H, W))

        # 计算更新后的特征
        update_feature = update_vertex * update_edge_converted

        # 学习偏移量
        offset = self.offset_conv(x)

        # 应用可变形卷积
        deformed_x = self.deform_conv(x, offset)

        # 添加残差连接
        deformed_x = deformed_x + x

        # 拼接可变形卷积后的特征和更新后的特征
        combined_feature = torch.cat((deformed_x, update_feature), dim=1)

        # 最终聚合
        output = self.final_aggregation_layer(combined_feature)
        return output


if __name__ == "__main__":
    # 设置输入张量大小
    batch_size = 1
    input_channels = 32
    height, width = 256, 256
    # 创建输入张量
    input_tensor = torch.randn(batch_size, input_channels, height, width)
    # 初始化 DAGEM 模块
    DAGEM = DAGEM(sync_bn=False, input_channels=input_channels)
    print(DAGEM)
    print("\n哔哩哔哩: CV缝合救星!\n")

    # 前向传播测试
    output = DAGEM(input_tensor)
    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
