import numpy as np
import pdb
import sklearn.metrics.pairwise as smp
import torch
import logging
# https://github.com/DistributedML/FoolsGold
# Takes in grad
# Compute similarity
# Get weightings
logger = logging.getLogger("logger")

def foolsgold(historical_weights):
    n_clients = len(historical_weights)
    grads = []
    clients_names = []
    epsilon = 1E-5

    for name, weight in historical_weights.items():
        grads.append(torch.flatten(weight).cpu().detach().numpy())
        clients_names.append(name)


    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1) #.astype(np.double)
    # print("maxcs", maxcs[ :20])

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))

    logger.info(f"wv1, {wv}")

    wv[wv > 1] = 1
    wv[wv < 0] = 0

    logger.info(f"wv2, {wv}")
    # if wv2 only has 0.
    if all(p == 0 for p in wv):
        wv = [1] * len(wv)
    else:
        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = 1 - epsilon

        logger.info(f"wv3, {wv}")

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        logger.info(f"wv4, {wv}")

    clients_weights = dict(zip(clients_names, wv))

    return clients_weights

# Incorporate reputation to the standard foolsgold
def foolsgold_contra(historical_weights, reputation_dict):
    n_clients = len(historical_weights)
    grads = []
    clients_names = []
    epsilon = 1E-5

    for name, weight in historical_weights.items():
        grads.append(torch.flatten(weight).cpu().detach().numpy())
        clients_names.append(name)

    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)  # .astype(np.double)
    # print("maxcs", maxcs[ :20])

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    if all(p == 0 for p in wv):
        wv = [1] * len(wv)
    else:
        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = 1 - epsilon

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

    clients_weights = dict(zip(clients_names, wv))

    for i in range(len(historical_weights)):
        _name = clients_names[i]
        if maxcs[i] >= 0.9:  # and i in bad_idx_loss:
            # wv[i] = 0
            reputation_dict[_name] = reputation_dict[_name] - 0.5
        else:
            reputation_dict[_name] = reputation_dict[_name] + 0.5

    return clients_weights, reputation_dict


# foolsgold from CONTRA
# https://github.com/SanaAwan5/CONTRA2020/
def foolsgold_new(historical_weights, reputation_dict):
    n_clients = len(historical_weights)
    # print("n_clients", n_clients)
    grads = []
    clients_names = []

    for name, weight in historical_weights.items():
        grads.append(torch.flatten(weight).cpu().detach().numpy())
        clients_names.append(name)

    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    avg_cs = []
    epsilon = 1E-5

    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / (maxcs[j] + epsilon)
    # k = 5 is not showing in the original Foolsgold
    k = 5
    losses =[]
    epsilon = 1E-5
    for i in range(n_clients):
        temp=0
        # k largest cs[i] in the good_idx
        good_idx = np.argpartition(cs[i], (n_clients - k))[(n_clients - k):n_clients]
        for j in range(k):
            good_idx_data = cs[i][good_idx[j]]
            temp += cs[i][good_idx[j]]
        # average of the k largest cs[i][j] in cs[i]
        temp = temp / k
        avg_cs.append(temp)

    avg_cs = np.array(avg_cs)
    maxcs = 1 - (np.max(cs, axis = 1))
    wv = 1 - (np.max(cs, axis=1))

    cs_sorted = np.sort(cs[:,-k:], axis = 0)
    cs_sorted_mean = (np.mean(cs_sorted, axis = 1))
    bad_idx_loss = np.argpartition(losses,-k)[-k:]
    for i in range(n_clients):
        if avg_cs[i] >=0.9 and i in bad_idx_loss:
            wv[i] = epsilon
        else:
            wv = (1 - (np.mean(cs_sorted, axis = 1))) + (np.mean(cs_sorted, axis = 1) - np.max(cs , axis = 1))
    wv[wv > (1 + epsilon)] = 1
    wv[wv < (0 + epsilon)] = 0

    alpha = np.max(cs, axis=1)

    if all(p == 0 for p in wv):
        wv = [1] * len(wv)
    else:
        # Rescale so that max value is wv
        wv = wv / (np.max(wv) + epsilon)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > (1 + epsilon))] = 1
        wv[(wv < (0 + epsilon))] = 0

    wv = [item * 100 for item in wv]
    clients_weights = dict(zip(clients_names, wv))

    for i in range(len(historical_weights)):
        _name = clients_names[i]
        if avg_cs[i] >= 0.9:  # and i in bad_idx_loss:
            # wv[i] = 0
            reputation_dict[_name] = reputation_dict[_name] - 0.5
        else:
            reputation_dict[_name] = reputation_dict[_name] + 0.5

    # wv is the weight
    return clients_weights, reputation_dict


# a fake aggregation rule to check the performance of foolsgold
def pendulum (historical_weights, count):
    n = len(historical_weights)
    clients_weights = dict()
    if count % 2 == 0:
       for key in historical_weights.keys():
           clients_weights[key] = 200 * (int(key[5:])%2)
    else:
        for key in historical_weights.keys():
            clients_weights[key] = 200 * ((int(key[5:]) + 1) % 2)

    return clients_weights