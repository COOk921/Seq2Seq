import torch
import pdb

def evaluate_accuracy(y_pred, y_label ):
    num = len(y_pred)
    all_acc = 0.0
    count = 0
    for i in range(num):
        pred = y_pred[i]
        label = y_label[i]
        acc = sum(pred == label) / len(pred)
        all_acc += acc
        count += 1
    return all_acc / count

def evaluate_pmr(y_pred, y_label, return_list=False):
    num = len(y_pred)
    all_acc = 0.0
    count = 0
    pmr_list = []
    for i in range(num):
        pred = y_pred[i]
        label = y_label[i]
        acc = 1 if sum(pred == label) == len(pred) else 0
        all_acc += acc
        pmr_list.append(acc)
        count += 1
    if return_list:
        return all_acc / count, pmr_list
    else:
        return all_acc / count

def evaluate_tau(y_pred, y_label, return_list=False):
    def kendall_tau(porder, gorder):
        pred_pairs, gold_pairs = [], []
        for i in range(len(porder)):
            for j in range(i+1, len(porder)):
                pred_pairs.append((porder[i], porder[j]))
                gold_pairs.append((gorder[i], gorder[j]))
        common = len(set(pred_pairs).intersection(set(gold_pairs)))
        uncommon = len(gold_pairs) - common
        tau = 1 - (2*(uncommon/len(gold_pairs)))
        
        return tau
    num = len(y_pred)
    all_tau = 0.0
    count = 0
    tau_list = []
    for i in range(num):
        pred = y_pred[i]
        label = y_label[i]
        if len(pred) == 1 and len(label) == 1:
            TAU = 1
        else:
            TAU = kendall_tau(pred, label)
            
        all_tau += TAU
        tau_list.append(TAU)
        count += 1
    if return_list:
        return all_tau / count, tau_list
    else:
        return all_tau / count


def main():
    torch.manual_seed(1234)
    # y_pred = torch.randint(0, 10, (2, 5))
    # y_label = torch.randint(0, 10, (2, 5))
    y_pred = torch.tensor([[1, 2, 3, 4, 5,6,7,8,9,10]])
    y_label = torch.tensor([[10, 2, 3, 4, 5,6,7,8,9,1]])


    acc = evaluate_accuracy(y_pred, y_label)
    pmr, pmr_list = evaluate_pmr(y_pred, y_label, return_list=True)
    tau, tau_list = evaluate_tau(y_pred.tolist(), y_label.tolist(), return_list=True)

    print(acc)
    print(pmr)
    print(tau)

if __name__ == "__main__":
    main()
