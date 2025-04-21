import torch

def calculate_tsp_tour_length(coords: torch.Tensor, path_indices: torch.Tensor) -> torch.Tensor:
    """
    根据给定的城市坐标和访问顺序索引，计算 TSP 旅程的总欧几里得距离。

    Args:
        coords (torch.Tensor): 城市坐标张量，形状为 [B, L, 2]，
                                B 是批次大小, L 是城市数量。
        path_indices (torch.Tensor): 代表访问顺序的城市索引张量，形状为 [B, L]。
                                     索引值应在 [0, L-1] 范围内。

    Returns:
        torch.Tensor: 包含批次中每个样本的总旅程长度的张量，
                      形状为 [B]。如果需要 [B, 1]，可以在返回后使用 .unsqueeze(1)。
    """
    # 获取批次大小和序列长度
    B, L, _ = coords.shape

    # 确保 path_indices 是 LongTensor 类型，以便用于 gather 操作
    path_indices = path_indices.long()

    # 扩展 path_indices 的维度以匹配坐标的最后一个维度 (2)
    # Shape: [B, L] -> [B, L, 2]
    expanded_path_indices = path_indices.unsqueeze(-1).expand(B, L, 2)

    # 使用 gather 根据 path_indices 重新排列坐标顺序
    # ordered_coords 的形状为 [B, L, 2]，包含了按访问顺序排列的城市坐标
    ordered_coords = torch.gather(coords, 1, expanded_path_indices)

    # 计算路径中连续城市之间的向量
    # current_cities: 城市 0 到 L-2 的坐标 -> Shape [B, L-1, 2]
    current_cities = ordered_coords[:, :-1, :]
    # next_cities: 城市 1 到 L-1 的坐标 -> Shape [B, L-1, 2]
    next_cities = ordered_coords[:, 1:, :]

    # 计算连续城市间的欧几里得距离 (段距离)
    # segment_distances 的形状为 [B, L-1]
    # ((next_cities - current_cities)**2).sum(dim=-1) 计算差向量平方和 (距离的平方)
    # torch.sqrt() 开方得到距离
    segment_distances = torch.sqrt(((next_cities - current_cities)**2).sum(dim=-1))

    # 计算从最后一个城市返回第一个城市的距离
    # last_city 的形状为 [B, 2]
    last_city = ordered_coords[:, -1, :]
    # first_city 的形状为 [B, 2]
    first_city = ordered_coords[:, 0, :]

    # return_distance 的形状为 [B]
    return_distance = torch.sqrt(((first_city - last_city)**2).sum(dim=-1))

    # 将所有段距离求和，并加上返回起点的距离，得到总旅程长度
    # total_length 的形状为 [B]
    total_length = segment_distances.sum(dim=1) + return_distance

    # 返回每个样本的总长度，形状为 [B]
    return total_length

if __name__ == '__main__':
    # 设置随机种子以便复现
    torch.manual_seed(42)

    # 定义批次大小和城市数量
    B, L = 4, 5 # Batch size 4, 5 cities

    # 生成随机城市坐标 (0到1之间)
    coords = torch.rand(B, L, 2)

    # 生成随机的访问路径 (确保每个路径都是 0 到 L-1 的一个排列)
    path_indices = torch.stack([torch.randperm(L) for _ in range(B)])

    # 计算总旅程长度
    tour_lengths = calculate_tsp_tour_length(coords, path_indices)

    # 打印结果
    print("示例坐标 (Batch 0):\n", coords[0])
    print("-" * 20)
    print("示例路径索引 (Batch 0):", path_indices[0])
    print("-" * 20)
    print("计算得到的总旅程长度:\n", tour_lengths)
    print("-" * 20)
    print("输出形状:", tour_lengths.shape) # 应该是 [B]

    # 如果需要 [B, 1] 的形状:
    # tour_lengths_unsqueezed = tour_lengths.unsqueeze(1)
    # print("unsqueeze(1)后的形状:", tour_lengths_unsqueezed.shape)