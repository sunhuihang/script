import torch
import torch.nn as nn


class GEM(nn.Module):
    def __init__(self, sync_bn=False, input_channels=256):
        super(GEM, self).__init__()
        self.input_channels = input_channels
        BatchNorm1d = nn.BatchNorm1d
        BatchNorm2d = nn.BatchNorm2d
        self.edge_aggregation_func = nn.Sequential(
            nn.Linear(4, 1),
            BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.vertex_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.edge_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.update_edge_reduce_func = nn.Sequential(
            nn.Linear(4, 1),
            BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
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
        edge = torch.stack(
            (
                torch.cat((input[:, :, -1:], input[:, :, :-1]), dim=2),
                torch.cat((input[:, :, 1:], input[:, :, :1]), dim=2),
                torch.cat((input[:, :, :, -1:], input[:, :, :, :-1]), dim=3),
                torch.cat((input[:, :, :, 1:], input[:, :, :, :1]), dim=3)
            ), dim=-1
        ) * input.unsqueeze(dim=-1)
        aggregated_edge = self.edge_aggregation_func(
            edge.reshape(-1, 4)
        ).reshape((B, C, H, W))
        cat_feature_for_vertex = torch.cat((vertex, aggregated_edge), dim=1)
        update_vertex = self.vertex_update_func(
            cat_feature_for_vertex.permute(0, 2, 3, 1).reshape((-1, 2 * self.input_channels))
        ).reshape((B, H, W, self.input_channels // 2)).permute(0, 3, 1, 2)
        # output = self.final_aggregation_layer(update_vertex)
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
        update_feature = update_vertex * update_edge_converted
        output = self.final_aggregation_layer(
            torch.cat((x, update_feature), dim=1)
        )
        return output


if __name__ == "__main__":
    # 设置输入张量大小
    batch_size = 1
    input_channels = 32
    height, width = 256, 256
    # 创建输入张量
    input_tensor = torch.randn(batch_size, input_channels, height, width)
    # 初始化 GEM 模块
    gem = GEM(sync_bn=False, input_channels=input_channels)
    print(gem)
    print("\n哔哩哔哩: CV缝合救星!\n")

    # 前向传播测试
    output = gem(input_tensor)
    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")