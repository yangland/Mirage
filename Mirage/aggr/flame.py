import torch
import numpy as np
import hdbscan
from copy import deepcopy
import torch.nn.functional as F
# from aggr.fltrust import modelsd2tensor
import sys
import os
from aggr.fedavg_median import fedavg

# from visualization import matrix_plot
# Manually specify the path to your project folder
project_path = os.path.abspath('..')  # Or provide the full path if needed
sys.path.append(project_path)
import logging
logger = logging.getLogger("individual_logging")
logger.addHandler(logging.StreamHandler())


def flame_aggr(server_model_state_dict, client_updates_dict, noise=0.001, exp_dir="", iter=0, **kwargs):
    client_ids = list(client_updates_dict.keys())
    reveled_model_dict = {}
    
    # Convert gradients to model states (current approach)
    for client_id in client_ids:
        reveled_model_dict[client_id] = get_model_merged(client_updates_dict[client_id], server_model_state_dict)
        
    # Get updated server state
    new_server_sd, flame_clients = flame(server_model_state_dict, reveled_model_dict, noise, client_ids, exp_dir, iter)
    
    # Return the gradient update (difference between new and old server state)
    server_grad = get_model_update(new_server_sd, server_model_state_dict)
    
    client_weights = {
        cid: (1.0 / len(flame_clients) if cid in flame_clients else 0.0)
        for cid in client_updates_dict
    }

    return server_grad, client_weights


def flame_old(server_sd, model_dict, noise=0.001, client_ids=[], exp_dir="", iter=0):
    
    cos_list=[]
    local_model_vector = []
    # caculate update_params(local gradients) from clients' sources and the target
    update_params = []
    client_ids = []
    num_clients = len(model_dict)
    server_sd_copy = deepcopy(server_sd)
    
        
    for client_id in sorted(model_dict.keys()):
        param = model_dict[client_id]
        local_model_vector.append(modelsd2flat(param))
        update_params.append(get_model_update(param, server_sd))
        client_ids.append(client_id)    
  
  
    for i in range(len(local_model_vector)):
        if torch.isnan(local_model_vector[i]).any():
            logger.info(f"FLAME: NaNs in local_model_vector[{i}], replacing with zeros")
            local_model_vector[i] = torch.nan_to_num(local_model_vector[i],
                                                     nan=0.0, 
                                                     posinf=0.0, 
                                                     neginf=0.0)
        
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - F.cosine_similarity(local_model_vector[i],
                                             local_model_vector[j],
                                             dim=0)
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)

    if np.isnan(cos_list).any():
        logger.info("FLAME: NaN detected in cos_list!")
        logger.info(f"FLAME: NaN locations: {np.where(np.isnan(cos_list))}")
        cos_list = np.nan_to_num(cos_list, nan=0.0)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    logger.info(f"FLAME: clusterer.labels_ {str(clusterer.labels_)}")
    benign_idx = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(num_clients):
            benign_idx.append(i)
            # norm_list = np.append(norm_list,torch.norm(modelsd2flat(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(num_clients):
            if clusterer.labels_[i] == max_cluster_index:
                benign_idx.append(i)
    
    
    # get the norm of all clients            
    for i in range(num_clients):
        flat = modelsd2flat(update_params[i])
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            print("FLAME: Tensor contains NaN or Inf values!")
            # You can handle it here, e.g., skip this update or replace NaNs
            flat = modelsd2flat(update_params[i])
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            norm = torch.norm(flat, p=2)
        else:
            norm = torch.norm(flat, p=2)
        norm_list = np.append(norm_list, norm.item())  # no consider BN

    if np.isnan(norm_list).any():
        norm_list = [val for val in norm_list if not np.isnan(val)] 
    
    if len(norm_list) == 0:
        raise ValueError("FLAME: norm_list is empty after NaN removal, cannot compute median.")
    
    # print(f"benign_idx: {benign_idx}")
    
    clip_value = np.median(norm_list)
    for i in range(len(benign_idx)):
        client_idx = benign_idx[i]
        gama = clip_value / (norm_list[client_idx] + 1e-15)
        if gama < 1:
            for key in update_params[client_idx]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[client_idx][key] *= gama

    logger.info(f"[FLAME DEBUG][Iter {iter}] Norm list: {norm_list}")
    logger.info(f"[FLAME DEBUG][Iter {iter}] Clip value: {clip_value}")
    
    # average of the model udpates then + the server model weight
    new_server_update, _ = fedavg(server_model_state_dict=server_sd_copy,
                               client_updates_dict={client_ids[i]: update_params[i] for i in benign_idx},
                               average_bn_buffers=True)
    new_server_sd = get_model_merged(new_server_update, server_sd_copy, assume_delta=False)
    
    #add noise
    with torch.no_grad():
        for key, var in new_server_sd.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                        continue
            temp = deepcopy(var)
            temp = torch.normal(mean=0, std=noise, size=var.shape).to(var.device)
            var += temp
    
    # convert the list index to the clients_ids
    benign_client_ids = [client_ids[i] for i in benign_idx]
    
    return  new_server_sd, benign_client_ids


def flame(server_sd, model_dict, noise=0.001, client_ids=None, exp_dir="", iter=0):

    # --- normalize server_sd to a plain state_dict ---
    if isinstance(server_sd, tuple):              # e.g., (state_dict, ...)
        server_sd = server_sd[0]
    if hasattr(server_sd, "state_dict"):          # nn.Module
        server_sd = server_sd.state_dict()
    # FIX: keep a detached, cloned copy (don't overwrite later)
    server_sd_copy = {k: v.detach().clone() for k, v in server_sd.items()}

    cos_list = []
    local_model_vector = []
    update_params = []
    local_client_ids = [] if client_ids is None else []  # we rebuild from model_dict
    num_clients = len(model_dict)

    # Build flattened models & updates
    for cid in sorted(model_dict.keys()):
        param_sd = model_dict[cid]     # dict of tensors (client model or delta dict)
        local_model_vector.append(modelsd2flat(param_sd))
        update_params.append(get_model_update(param_sd, server_sd))  # DIFF = client - server
        local_client_ids.append(cid)

    # Sanitize local_model_vector
    for i in range(len(local_model_vector)):
        if torch.isnan(local_model_vector[i]).any() or torch.isinf(local_model_vector[i]).any():
            logger.info(f"FLAME: NaNs/Infs in local_model_vector[{i}], replacing with zeros")
            local_model_vector[i] = torch.nan_to_num(local_model_vector[i], nan=0.0, posinf=0.0, neginf=0.0)

    # Pairwise cosine distance matrix for HDBSCAN (NumPy array)
    for i in range(len(local_model_vector)):
        row = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - F.cosine_similarity(local_model_vector[i], local_model_vector[j], dim=0)
            row.append(cos_ij.item())
        cos_list.append(row)
    cos_mat = np.asarray(cos_list, dtype=np.float64)
    if np.isnan(cos_mat).any():
        logger.info("FLAME: NaN detected in cos matrix; replacing with zeros")
        cos_mat = np.nan_to_num(cos_mat, nan=0.0)

    # Cluster to pick benign subset
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=num_clients // 2 + 1,
        min_samples=1,
        allow_single_cluster=True
    ).fit(cos_mat)

    logger.info(f"FLAME: clusterer.labels_ {str(clusterer.labels_)}")

    benign_idx = []
    if clusterer.labels_.max() < 0:
        benign_idx = list(range(num_clients))
    else:
        # largest cluster
        labels = clusterer.labels_
        max_cluster = None
        max_size = -1
        for c in range(labels.max() + 1):
            size = (labels == c).sum()
            if size > max_size:
                max_size = size
                max_cluster = c
        benign_idx = [i for i in range(num_clients) if labels[i] == max_cluster]

    # Norm list & clipping
    norm_list = []
    for i in range(num_clients):
        flat = modelsd2flat(update_params[i])
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            logger.warning("FLAME: NaN/Inf in update; replacing with zeros for norm calc")
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        norm_list.append(flat.norm(p=2).item())
    norm_list = np.array(norm_list, dtype=np.float64)

    if np.isnan(norm_list).any():
        norm_list = norm_list[~np.isnan(norm_list)]
    if norm_list.size == 0:
        raise ValueError("FLAME: norm_list is empty after NaN removal")

    clip_value = np.median(norm_list)
    for idx in benign_idx:
        gamma = clip_value / (norm_list[idx] + 1e-15)
        if gamma < 1.0:
            for k, t in update_params[idx].items():
                if k.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[idx][k] = t * gamma

    logger.info(f"[FLAME DEBUG][Iter {iter}] Norm list: {norm_list}")
    logger.info(f"[FLAME DEBUG][Iter {iter}] Clip value: {clip_value}")

    # Average deltas of benign subset
    avg_update, _ = fedavg(
        server_model_state_dict=server_sd_copy,
        client_updates_dict={local_client_ids[i]: update_params[i] for i in benign_idx},
        average_bn_buffers=True
    )

    # FIX: avg_update is a DELTA dict -> merge as delta
    new_server_sd = get_model_merged(avg_update, server_sd_copy, assume_delta=True)

    # NOTE: never corrupt BN buffers with noise; only apply noise to weights/biases
    with torch.no_grad():
        for key, var in new_server_sd.items():
            suffix = key.split('.')[-1]
            if suffix in ('num_batches_tracked',):
                continue
            if ('running_mean' in key) or ('running_var' in key):
                continue  # <-- DO NOT NOISE BN buffers
            # small noise
            var.add_(torch.normal(mean=0.0, std=noise, size=var.shape, device=var.device))

    # Extra safety: ensure BN running_var stays positive
    with torch.no_grad():
        for key, var in new_server_sd.items():
            if 'running_var' in key:
                var.clamp_(min=1e-6)

    benign_client_ids = [local_client_ids[i] for i in benign_idx]
    return new_server_sd, benign_client_ids



def get_model_update(updated_model, model):
    """
    Returns the difference between updated_model and model (server),
    while preserving 'num_batches_tracked' and other non-learnable buffers.
    """
    update = {}
    for key in updated_model:
        if key.endswith('num_batches_tracked'):
            # Preserve the original value from the server model
            update[key] = model[key].detach().clone()
        else:
            update[key] = updated_model[key] - model[key].detach()
    return update


def get_model_merged(gradient_update, base_model, assume_delta=True):
    # normalize base_model to a state_dict
    if isinstance(base_model, tuple):
        base_model = base_model[0]
    if hasattr(base_model, "state_dict"):
        base_sd = base_model.state_dict()
    elif isinstance(base_model, dict):
        base_sd = base_model
    else:
        raise TypeError(f"Unsupported base_model type: {type(base_model)}")

    merged = {}
    for key, base_tensor in base_sd.items():
        if key.endswith('num_batches_tracked'):
            merged[key] = base_tensor.detach().clone()
        else:
            if assume_delta:
                merged[key] = base_tensor.detach() + gradient_update[key]
            else:
                merged[key] = gradient_update[key]  # already absolute
    return merged


def modelsd2flat(model_dict):
    """
    Flattens a model's state dictionary into a single 1D tensor.
    Skips non-trainable buffers like 'num_batches_tracked' for consistency.
    """
    ravel_list = []
    for layer_name, parms in model_dict.items():
        if isinstance(parms, torch.Tensor) and not layer_name.endswith('num_batches_tracked'):
            ravel_list.append(parms.detach().reshape(-1))
    
    if not ravel_list:
        return torch.tensor([])
    
    flat_tensor = torch.cat(ravel_list)
    return flat_tensor