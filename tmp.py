import torch
import torch.nn as nn
import torch.nn.functional as F



# 模型输出的对数概率
probs = torch.tensor( [[0.0, 0.1, 0.9],
                        [0.1, 0.4, 0.5]] )
label = torch.tensor([2, 2])
print(probs.shape)



log_lable = F.log_softmax(probs, dim=-1)
print(log_lable)
loss = nn.NLLLoss()(log_lable, label)
print(loss.item())
print("NLLLoss: ", loss.item())



# 创建 NLLLoss 实例
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(probs, label)

print(f"计算得到的 CrossEntropyLoss: {loss.item():.4f}")