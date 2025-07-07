import torch
from aggr.fltrust import modelsd2tensor
from aggr.fedavg_median import fedavg
import math
from copy import deepcopy

def krum_aggr(model_dict, f, m):
    client_ids = list(model_dict.keys())
    reveled_model_dict = {}
    
    for client_id in client_ids:
        reveled_model_dict[client_id] = modelsd2tensor(model_dict[client_id])

    selected_clients_with_scores, krum_clients, client_weights = \
        krum(reveled_model_dict, f, m)
    
    krum_clients.sort()
    krum_model_dict = dict((client_id, model_dict[client_id]) for client_id in krum_clients)
    new_weights = average_tensor_dict(krum_model_dict)
    
    return new_weights, krum_clients, selected_clients_with_scores


def average_tensor_dict(tensor_dicts):
    """Averages dicts of gradients (same format as state_dicts)"""
    tensor_list = list(tensor_dicts.values())
    avg = deepcopy(tensor_list[0])
    for k in avg:
        avg[k] = torch.stack([t[k].float() for t in tensor_list]).mean(0)
    return avg

# Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Peva Blanchard)
# https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
def krum(model_sample_weight_dict, f, m):
    n = len(model_sample_weight_dict)
    if f >= math.floor(n / 2) - 1:
        f = math.floor(n / 2) - 2
    assert f < (n / 2 - 1), "f should be less than n/2 - 1"
    if m >= n:
        m = n - 1
    assert 0 <= m <= n, "m should >=0 and <= n"
    # if m == 1, original krum; m > 1 multi-krum; m == n, FedAvg

    # get krum scores
    krum_scores = {}
    l2_dist = {k: [] for k in model_sample_weight_dict.keys()}
    for name1, value1 in model_sample_weight_dict.items():
        for name2, value2 in model_sample_weight_dict.items():
            if name1 != name2:
                dist = torch.norm(value1 - value2, p=2).item()
                dist = torch.cdist(value1, value2, p=2)
                l2_dist[name1].append(dist.item())

    for name, dists in l2_dist.items():
        # sum of first n-f-2 distances
        # print('dists', len(dists))
        sorted_dist = dists
        sorted_dist.sort()
        # print("sorted_dist sum", sum(sorted_dist))
        # print("n, f", n, f)
        krum_score = sum(sorted_dist[:n - f - 2])
        # print("krum_score", krum_score)
        krum_scores[name] = krum_score

    sorted_krum_scores = sorted(krum_scores.items(), key=lambda x: x[1])

    # print("sorted_krum_scores", sorted_krum_scores)

    selected_clients_with_scores = sorted_krum_scores[:m]
    krum_clients = [i[0] for i in selected_clients_with_scores]
    total_weights = n * 100
    ind_weight = total_weights / m

    client_weights = dict.fromkeys(model_sample_weight_dict.keys(), 0)

    for name, weight in client_weights.items():
        if name in krum_clients:
            client_weights[name] = ind_weight
        else:
            client_weights[name] = 0

    return selected_clients_with_scores, krum_clients, client_weights
