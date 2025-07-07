import torch
import numpy as np
import hdbscan
from copy import deepcopy
import torch.nn.functional as F
# from aggr.fltrust import modelsd2tensor
import sys
import os
from utilities import get_model_update, no_defence_balance, get_model_merged, get_model_update, modelsd2flat
from aggr.fedavg_median import fedavg
from visualization import matrix_plot
# Manually specify the path to your project folder
project_path = os.path.abspath('..')  # Or provide the full path if needed
sys.path.append(project_path)
import logging
logger = logging.getLogger("individual_logging")
logger.addHandler(logging.StreamHandler())


def flame_aggr(server_sd, grad_dict, noise=0.001, exp_dir="", iter=0):
    client_ids = list(grad_dict.keys())
    reveled_model_dict = {}
    
    # Convert gradients to model states (current approach)
    for client_id in client_ids:
        reveled_model_dict[client_id] = get_model_merged(grad_dict[client_id], server_sd)
        
    # Get updated server state
    new_server_sd, flame_clients = flame(server_sd, reveled_model_dict, noise, client_ids, exp_dir, iter)
    
    # Return the gradient update (difference between new and old server state)
    server_grad = get_model_update(new_server_sd, server_sd)
    
    return server_grad, flame_clients


def flame(server_sd, model_dict, noise=0.001, client_ids=[], exp_dir="", iter=0):
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
    
    # logger.info(f"cos_list len: {len(cos_list)}")
    # logger.info(f"cos_M: {np.array(cos_list)}")
    
    # plot the cosine similarity matrix
    # Create the directory if it doesn't exist
    os.makedirs(f"{exp_dir}/plots/flame_cos", exist_ok=True)

    # Plot the cosine similarity matrix
    matrix_plot(
        np.array(cos_list), 
        f"cos_M{iter}", 
        f"cos_M_{iter}", 
        "client_id", 
        "client_id",
        f"{exp_dir}/plots/flame_cos", 
        client_ids=client_ids
    )
    
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
    # new_server_sd = no_defence_balance([update_params[i] for i in benign_idx], server_sd_copy)
    new_server_update = fedavg([update_params[i] for i in benign_idx])
    new_server_sd = get_model_merged(new_server_update, server_sd_copy)
    
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

