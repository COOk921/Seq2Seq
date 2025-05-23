from matplotlib import pyplot as plt
import torch
import numpy as np

def show_tsp_data(input,label):
    cities = input.numpy()
    route = label.numpy()

    route = np.append(route, route[0])
    oute = np.append(route, route[0])

    plt.figure(figsize=(10, 8))


    plt.scatter(cities[:, 0], cities[:, 1], c='black', s=100, label='Cities')

    # 根据路线绘制旅行路径
    route_cities = cities[route]
    plt.plot(route_cities[:, 0], route_cities[:, 1], 'r-', linewidth=2, label='Route')

    # 添加起点标记
    plt.scatter(route_cities[0, 0], route_cities[0, 1], c='green', s=350, marker='*', label='Start')


    plt.title('TSP Route')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    plt.show()

loaded_data = torch.load('data/tsp/tsp_data.pt')
input = loaded_data['input']  # [B, L, 2]
label = loaded_data['label']  # [B, L]


sample_idx = 1000
cities = input[sample_idx]  
route = label[sample_idx]   

show_tsp_data(cities,route)






