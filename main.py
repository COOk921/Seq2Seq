import torch
import torch.nn as nn
from model.pointer_network import PointerNetwork
from data.container_dataset import ContainerDataset
from data.tsp import TSPDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from evaluate import evaluate_accuracy, evaluate_pmr, evaluate_tau
from utils.optim import build_optimizer
from utils.loss import binary_listNet, sorting_loss
from utils.show_tsp import show_tsp_data
from utils.align import align_label_start
from utils.tourLen import calculate_tsp_tour_length

import numpy as np
import time
import math
import pdb


def train_model(model, train_dataloader, test_dataloader, data_size, epochs, lr, device):

    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.NLLLoss().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_tau = 999
    best_epoch = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        epoch_loss = 0

        for batch in train_dataloader:
            model.train()
            input = batch['input']
            label = batch['label'].squeeze(-1).long()
            outputs, pointers = model(input)
           
            align_label,err= align_label_start(label, pointers)  # 调整标签顺序       
            if err:
                continue
            
            loss = criterion(outputs, align_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_dataloader)}")

        if epoch % 2 == 0 and epoch != 0:  # 每隔一个 epoch 进行测试
            tau = test_model(model, test_dataloader, data_size)
            if tau < best_tau:
                best_tau = tau
                torch.save(model.state_dict(), f'checkpoint/sort_best_model_for{data_size}.pth')
                best_epoch = epoch + 1
                print(f"Best tau at epoch {best_epoch}: {best_tau:.4f}...save model.")
            scheduler.step(tau)

    print(f"Best avg_dis at epoch {best_epoch}: {best_tau:.4f}")

def test_model(model, test_dataloader, data_size):

    all_outputs = []
    all_labels = []
    all_tour_lengths = []
    

    print("test model...")
    for batch in test_dataloader:
        model.eval()
        with torch.no_grad():
            input = batch['input']
            label = batch['label'].squeeze(-1).long()
            output, pointer = model(input)
            align_label,err = align_label_start(label, pointer)


            tour_lengths = calculate_tsp_tour_length(input,align_label) 
            tour_lengths2 = calculate_tsp_tour_length(input,pointer)

            dis = tour_lengths2 - tour_lengths
            avg_dis = torch.mean(dis)

            all_tour_lengths.append(avg_dis.cpu().numpy())

            if err:
                continue
            #show_tsp_data(input[0], pointer[0],align_label[0])  # 显示第一个样本的预测标签

            all_outputs.append(pointer)
            all_labels.append(align_label)
    
    avg_dis = np.mean(all_tour_lengths)  # 计算平均距离

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算多种评估指标
    acc = evaluate_accuracy(all_outputs, all_labels)
    pmr = evaluate_pmr(all_outputs, all_labels)
    tau = evaluate_tau(all_outputs.tolist(), all_labels.tolist())

    
    print(f"Accuracy: {acc:.4f}")
    print(f"Perfect Match Rate: {pmr:.4f}")
    print(f"Kendall Tau: {tau:.4f}")
    print(f"Average distance: {avg_dis:.4f}")
    print("--------------------------------")

    return avg_dis

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model

if __name__ == "__main__":

    epochs = 100
    lr = 1e-4
    data_size = 15  # 15个城市需要排序
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointerNetwork(
        elem_dims=2,  # 初始输入维度
        embedding_dim=256,  # 增加嵌入维度
        hidden_dim=256,  # 增加LSTM的维度
        lstm_layers=1,  # 增加LSTM层数
        dropout=0.2,
        bidir=False,  # 使用双向LSTM
        masking=True,
        output_length=data_size,
    ).to(device)

    train_dataset = TSPDataset(size=data_size, type='train', seed=4412)
    test_dataset = TSPDataset(size=data_size, type='test', seed=4412)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 启用shuffle
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_model(model, train_dataloader, test_dataloader, data_size, epochs, lr, device)

    model = load_model(model,f'checkpoint/sort_best_model_for{data_size}.pth')
    test_model(model, test_dataloader, data_size)

    torch.cuda.empty_cache()