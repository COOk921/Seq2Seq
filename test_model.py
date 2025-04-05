import torch
import torch.nn as nn
from model.pointer_network import PointerNetwork
from data.tsp_dataset import SortingDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from evaluate import evaluate_accuracy, evaluate_pmr, evaluate_tau
from utils.optim import build_optimizer
from utils.loss import binary_listNet,sorting_loss
import math
import pdb

def train_model(model, train_dataloader, test_dataloader, data_size, epochs, lr,device):

    

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 使用学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        epoch_loss = 0
        
        for batch in train_dataloader:
            model.train()
            input = batch['input']
            label = batch['label'].squeeze(-1).long()
            outputs, pointers = model(input)
            
            # 使用自定义排序损失函数
            #loss = sorting_loss(pointers, label).requires_grad_(True)
            
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_dataloader)}")
        if epoch % 2 and epoch != 0:
            acc = test_model(model,test_dataloader,data_size)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f'checkpoint/sort_best_model_for{data_size}.pth')
                best_epoch = epoch + 1
                print(f"Best tau at epoch {best_epoch}: {best_acc}...save model.")
    
    print(f"Best tau at epoch {best_epoch}: {best_acc}")

def test_model(model, test_dataloader,data_size):

    all_outputs = []
    all_labels = []

    print("test model...")
    for batch in test_dataloader:
        model.eval()
        with torch.no_grad():
            input = batch['input']
            label = batch['label'].squeeze(-1).long()
            
            output, pointer = model(input)
            all_outputs.append(pointer)
            all_labels.append(label)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算多种评估指标
    acc = evaluate_accuracy(all_outputs, all_labels)
    pmr = evaluate_pmr(all_outputs, all_labels)
    tau = evaluate_tau(all_outputs.tolist(), all_labels.tolist())
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Perfect Match Rate: {pmr:.4f}")
    print(f"Kendall Tau: {tau:.4f}")
    print("--------------------------------")
    
    # 返回Kendall Tau作为主要评估指标
    return tau

def load_model(model,checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model

if __name__ == "__main__":
    
    epochs = 30
    lr = 1e-4
    data_size = 20
    dim = math.ceil(math.log2(data_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    print(f"data_size: {data_size}, dim: {dim}")
    model = PointerNetwork(
        elem_dims = dim,      #初始输入维度
        embedding_dim=256,  # 增加嵌入维度
        hidden_dim= 128,   # 增加LSTM的维度
        lstm_layers=1,    # 增加LSTM层数
        dropout=0.1,  
        bidir=False,  # 使用双向LSTM
        masking=True,
        # output_length=30,
    ).to(device)

    train_dataset = SortingDataset(size=data_size, num_samples=500,seed=42)
    test_dataset = SortingDataset(size=data_size, num_samples=100,seed=42)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 启用shuffle
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_model(model, train_dataloader, test_dataloader, data_size, epochs, lr,device) 

    # model = load_model(model,f'checkpoint/sort_best_model_for{data_size}.pth')
    # test_model(model, test_dataloader, data_size)

    torch.cuda.empty_cache()
    
