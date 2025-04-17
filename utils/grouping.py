import torch
import pdb
import numpy as np
import time
from sklearn.cluster import KMeans,Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# group_size * group_num = max_len
group_size = 1
group_num = 20

def trans_to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def grouping(casEmbed, cas_mask):

    batch_size, seq_len, embed_size = casEmbed.size()
    casEmbed = casEmbed * ( ~cas_mask.unsqueeze(-1) ) 
   
    all_labels = []
    count = []                  #每个group中的有效长度
    for b in range(batch_size):
        batch_data = casEmbed[b]
        batch_mask = cas_mask[b] 

        max_len = torch.sum(~batch_mask).item() # 有效长度
        k = max_len // group_size  
        k = 2 if k < 2 else k          # 至少要分两组
    
        labels_b = cluster(batch_data, k)
        all_labels.append(labels_b)

        group_count = torch.bincount(labels_b)  # 统计每个group的长度
        group_count = torch.cat([group_count, torch.zeros(group_num - len(group_count), dtype=torch.long, device=casEmbed.device)])
        count.append(group_count)
    
    count = torch.stack(count, dim=0)
    
    grouped_labels = torch.stack(all_labels, dim=0)
    grouped_labels = trans_to_cuda(grouped_labels)

    count = torch.tensor(count, device=casEmbed.device) #[batch, group_num]

    num_groups = torch.max(grouped_labels).item() + 1
    
    groupEmbed = torch.zeros(
        batch_size, 
        num_groups, 
        embed_size, 
        device=casEmbed.device
    )
    
    index = grouped_labels.unsqueeze(-1).expand(-1, -1,embed_size).to(torch.long)
   
    groupEmbed.scatter_add_(1, index, casEmbed)

    groupEmbed = torch.nn.functional.pad(groupEmbed, (0, 0, 0, group_num - num_groups))
    group_mask = torch.all(groupEmbed == 0, dim=2)

    return groupEmbed, group_mask, count

def cluster(X, k ):

    """ CPU 归一化 """
    # X_cpu = X.cpu().detach().numpy()
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # X_norm = scaler.fit_transform(X_cpu )

    """ GPU 归一化 """
    min_vals = torch.min(X, dim=1, keepdim=True)[0]
    max_vals = torch.max(X, dim=1, keepdim=True)[0]
    denominator = max_vals - min_vals
    zero_denominator_mask = denominator == 0
   
    denominator = torch.where(zero_denominator_mask, torch.tensor(1e-8, dtype=torch.long,device = X.device), denominator)
    X_norm = (X - min_vals) / denominator
    X_norm = torch.where(zero_denominator_mask, torch.zeros_like(X), X_norm)
    
    """ Birch """
    # birch = Birch(n_clusters=k)
    # birch.fit(X_norm)
    # labels = torch.from_numpy(birch.labels_)

    """ sklearn的KMeans(CPU) """
    # kmeans = KMeans(
    #     n_clusters=k,
    #     n_init=2,       # 重复2次，取最好的结果
    #     max_iter=20,
    #     tol=1e-3,
    #     )
    # kmeans.fit(X_norm)
    # labels = torch.from_numpy(kmeans.labels_)
    
    """ 自己实现的KMeans(GPU) """
    centroids, labels = gpu_friendly_kmeans(X_norm, k)

    #pdb.set_trace()
    return labels

def gpu_friendly_kmeans(X, k, max_iters = 20, tol=1e-3, device='cuda' ):
   
    n_samples, n_features = X.shape
    
    # 随机初始化 k 个质心
    indices = torch.randperm(n_samples)[:k]
    centroids = kmeans_plus_plus_init(X, k) #

    for iter_num in range(max_iters):
            # 计算每个数据点与各个质心之间的平方距离
            squared_X = (X ** 2).sum(dim=1, keepdim=True)             
            squared_centroids = (centroids ** 2).sum(dim=1).unsqueeze(0)  
            distances = squared_X + squared_centroids - 2 * torch.mm(X, centroids.t())
           
            labels = torch.argmin(distances, dim=1) 
            new_centroids = torch.zeros((k, n_features), device=device)
            counts = torch.zeros(k, device=device)

            new_centroids = new_centroids.index_add(0, labels, X)
           
            ones = torch.ones(n_samples, device=device)
            counts = counts.index_add(0, labels, ones)
            counts_unsqueezed = counts.unsqueeze(1) 
            mask = (counts != 0) 

            # 对非空聚类进行归一化（广播时 counts_unsqueezed[mask] 的形状为 (num_valid, 1)）
            new_centroids[mask] = new_centroids[mask] / counts_unsqueezed[mask]
            
            # 对于空聚类，将新的质心设置为原来的质心（也可以选择重新随机初始化）
            empty_clusters = (counts == 0)
            if empty_clusters.any():
                new_centroids[empty_clusters] = centroids[empty_clusters]

            # 判断是否收敛：所有质心移动的距离都小于 tol
            centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
            if centroid_shift < tol:
                centroids = new_centroids
                break

            centroids = new_centroids

    return centroids, labels
def kmeans_plus_plus_init(X, k):
    n_samples, n_features = X.shape
    centroids = torch.zeros((k, n_features), device=X.device)
    # 随机选择第一个质心
    first_centroid_index = torch.randint(0, n_samples, (1,)).item()
    centroids[0] = X[first_centroid_index]

    for i in range(1, k):
        # 计算每个样本到已选质心的最小距离的平方
        distances = torch.cdist(X, centroids[:i])
        min_distances_squared = torch.min(distances, dim=1)[0] ** 2

        epsilon = 1e-8
        probabilities = min_distances_squared / (torch.sum(min_distances_squared) + epsilon)

        if probabilities.sum() == 0:
            probabilities = torch.ones_like(probabilities) / len(probabilities)

        next_centroid_index = torch.multinomial(probabilities, 1).item()
        centroids[i] = X[next_centroid_index]

    return centroids