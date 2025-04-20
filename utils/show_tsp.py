import matplotlib.pyplot as plt
import numpy as np
import torch

def show_tsp_data(input, predicted_label, true_label=None):
    """
    绘制 TSP 数据，包括城市分布和预测/真实路径。

    Args:
        input (torch.Tensor): 形状为 [L, 2] 的张量，表示城市的坐标。
        predicted_label (torch.Tensor): 形状为 [L] 的张量，表示预测的城市访问顺序。
        true_label (torch.Tensor, 可选): 形状为 [L] 的张量，表示真实的城市访问顺序。
                                      如果提供，则绘制真实路径；否则只绘制预测路径。
    """
    cities = input.cpu().numpy()
    predicted_route = predicted_label.cpu().numpy()

    predicted_route = np.append(predicted_route, predicted_route[0]) #closed loop

    plt.figure(figsize=(10, 5))  # 调整图形大小以容纳两个子图
    # 绘制城市分布
    plt.subplot(1, 2, 1)
    plt.scatter(cities[:, 0], cities[:, 1], c='black', s=100, label='Cities')
   
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    # 绘制预测路径
    predicted_route_cities = cities[predicted_route]
    plt.plot(predicted_route_cities[:, 0], predicted_route_cities[:, 1], 'r-', linewidth=2, label='Predicted Route')
    plt.scatter(predicted_route_cities[0, 0], predicted_route_cities[0, 1], c='green', s=350, marker='*', label='Start') #start point
    plt.legend()

    if true_label is not None:
        true_route = true_label.cpu().numpy()
        true_route = np.append(true_route, true_route[0])
        # 绘制真实路径
        plt.subplot(1, 2, 2)
        plt.scatter(cities[:, 0], cities[:, 1], c='black', s=100, label='Cities')
        true_route_cities = cities[true_route]
        plt.plot(true_route_cities[:, 0], true_route_cities[:, 1], 'b-', linewidth=2, label='True Route')
        plt.scatter(true_route_cities[0, 0], true_route_cities[0, 1], c='green', s=350, marker='*', label='Start')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
    else:
        plt.subplot(1, 2, 2)
        plt.scatter(cities[:, 0], cities[:, 1], c='black', s=100, label='Cities')
        plt.plot(predicted_route_cities[:, 0], predicted_route_cities[:, 1], 'r-', linewidth=2, label='Predicted Route')
        plt.scatter(predicted_route_cities[0, 0], predicted_route_cities[0, 1], c='green', s=350, marker='*',label='Start')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()  # 调整子图布局，避免重叠
    plt.show()