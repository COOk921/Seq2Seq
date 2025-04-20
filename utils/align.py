import torch
import pdb

def align_label_start(label, pointer):
    """
    调整标签 (label) 的顺序，使其与预测 (pointer) 的起始城市一致。

    Args:
        label (torch.Tensor): 形状为 [B, L] 的标签，表示真实的城市访问顺序。
        pointer (torch.Tensor): 形状为 [B, L] 的预测，表示模型预测的城市访问顺序。

    Returns:
        torch.Tensor: 调整后的标签，形状为 [B, L]。
    """
    aligned_labels = []
    err = False
    
    for b in range(label.size(0)):
        label_b = label[b]
        pointer_b = pointer[b]
        try:
            start_index = (label_b == pointer_b[0]).nonzero(as_tuple=True)[0][0]
            aligned_label_b = torch.cat((label_b[start_index:], label_b[:start_index]))
        except IndexError:
            # print(f"Warning: No matching start city found for batch {b}. Skipping alignment.")
            # pdb.set_trace()
            aligned_labels.append(pointer_b)  #  标签存在错误，直接使用预测的标签

            err = True
            continue  

        aligned_labels.append(aligned_label_b)
    return torch.stack(aligned_labels), err  

# # 示例
# label = torch.tensor([[ 0,  5,  4,  2, 11, 10, 13,  7,  9,  6, 12,  8,  1, 14,  3],
#                        [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 0]], device='cuda:0')
# pointer = torch.tensor([[ 3, 13, 10,  7,  6,  9, 11,  2,  4, 12,  5,  8,  1,  0, 14],
#                          [5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 0,  1,  2,  3, 4]], device='cuda:0')

# aligned_label = align_label_start(label, pointer)
# print("Original label:", label)
# print("Aligned label:", aligned_label)