import torch
import torch.nn.functional as F
import pdb

PADDED_Y_VALUE = -1
DEFAULT_EPS = 1e-8

def binary_listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListNet loss variant for binary ground truth data introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0
    normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
    normalizer[normalizer == 0.0] = 1.0
    normalizer = normalizer.expand(-1, y_true.shape[1])
    y_true = torch.div(y_true, normalizer)

    preds_smax = F.softmax(y_pred, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(y_true * preds_log, dim=1))

def sorting_loss(predictions, targets):
    """
    自定义排序损失函数，用于排序任务
    :param predictions: 模型预测的排序索引，shape [batch_size, seq_length]
    :param targets: 目标排序索引，shape [batch_size, seq_length]
    :return: 损失值
    """
    batch_size, seq_length = predictions.shape
    
    # 计算预测和目标之间的差异
    diff = predictions - targets
    
    # 计算绝对误差
    abs_error = torch.abs(diff).float()
    
    # 计算相对位置误差（考虑相邻元素之间的相对顺序）
    relative_error = torch.zeros_like(abs_error)
    for i in range(seq_length - 1):
        for j in range(i + 1, seq_length):
            # 检查相对顺序是否正确
            pred_order = predictions[:, i] < predictions[:, j]
            target_order = targets[:, i] < targets[:, j]
            relative_error[:, i] += (pred_order != target_order).float()
            relative_error[:, j] += (pred_order != target_order).float()
    
    # 组合绝对误差和相对误差
    loss = torch.mean(abs_error) + 0.5 * torch.mean(relative_error)
    
    return loss