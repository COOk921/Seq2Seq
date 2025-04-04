from model.encoder import Encoder
from model.sortTransfoemer import SortingTransformer
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pdb

N, M, D = 5000, 10, 4   #[B, L, D]

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    X = np.zeros((N, M, D), dtype=np.float32)
    Y = np.zeros((N, M), dtype=np.int64)
    for i in range(N):
        random_numbers = np.random.choice(10, M, replace=False)
        for j in range(M):
            binary_representation = list(map(int, np.binary_repr(random_numbers[j], width=D)))
            X[i, j, :] = binary_representation
            Y[i, j] = random_numbers[j]
    
    tensor_X = torch.tensor(X, dtype=torch.float32).to(device)
    tensor_Y = torch.tensor(Y, dtype=torch.long).to(device)

    dataset = TensorDataset(tensor_X, tensor_Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SortingTransformer(
        input_dim=D,    
        d_model=128,
        num_heads=4,
        num_layers=3,
        output_dim=M
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        total_loss = 0
        for batch_X, batch_Y in dataloader:
            batch_size,seq_len,_=batch_X.shape
            mask = torch.zeros(batch_size,seq_len,seq_len,dtype=torch.bool).to(device)
            optimizer.zero_grad()
            outputs = []

            for i in range(seq_len):
                logits = model(batch_X, mask) # [B, M, M]
                pred = torch.argmax(logits, dim=-1)
                
                 mask.scatter_(1, pred.unsqueeze(1), True)
                
                outputs.append(pred)
                
            outputs = torch.stack(outputs, dim=1)
            
            pdb.set_trace()
            loss = criterion(outputs.permute(0, 2, 1), batch_Y)  # Cross entropy expects (N, C, M)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        if epoch % 3 == 0:
            torch.save(model.state_dict(), "model.pth")
            test()

  
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SortingTransformer(
        input_dim=D,    
        d_model=128,
        num_heads=4,
        num_layers=3,
        output_dim=M
    ).to(device)

    model.load_state_dict(torch.load("model.pth"))

    test_X = np.zeros((1, M, D), dtype=np.float32)

    random_numbers = np.random.choice(10, M, replace=False)
    for j in range(M):
        binary_representation = list(map(int, np.binary_repr(random_numbers[j], width=D)))
        test_X[0, j, :] = binary_representation
    print("test_X:{}".format(random_numbers))
   
    test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    outputs = model(test_tensor)
    outputs = outputs.softmax(dim=-1)
    outputs = torch.argmax(outputs, dim=-1)
    print("outputs:{}".format(outputs))

if __name__ == "__main__":
    train()
