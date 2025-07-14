import numpy as np
import torch
from copy import deepcopy

def coordinate_wise_median(server_model_state_dict, client_updates_dict, **kwargs):
    new_weights = dict()
    client_ids = list(client_updates_dict.keys())

    with torch.no_grad():
        for name, param in client_updates_dict[client_ids[0]].items():
            layer_weights = []
            for client_id in client_ids:
                layer_weights.append(client_updates_dict[client_id][name])

            t = torch.stack(layer_weights)
            med = t.float().median(0)[0]
            new_weights[name] = med
    
    return new_weights


# def fedavg(server_model_state_dict, client_updates_dict, average_bn_buffers=True, **kwargs):
#     """
#     Federated averaging of model parameters.
    
#     Args:
#         models: Dict or list of model state_dicts
#         average_bn_buffers: Whether to average BatchNorm buffers (running_mean/var)
#                            num_batches_tracked is always taken from first model
#     Returns:
#         Averaged state_dict
#     """
#     model_list = list(client_updates_dict.values()) if isinstance(client_updates_dict, dict) else client_updates_dict
#     avg_model = deepcopy(model_list[0])

#     # Process all parameters except num_batches_tracked
#     for key in avg_model:
#         if 'num_batches_tracked' in key:
#             continue
            
#         if any(buf in key for buf in ['running_mean', 'running_var']):
#             if average_bn_buffers:
#                 avg_model[key] = torch.stack([m[key] for m in model_list]).mean(0)
#             continue
            
#         avg_model[key] = torch.stack([m[key] for m in model_list]).mean(0)

#     return avg_model


def fedavg(server_model_state_dict, client_updates_dict, average_bn_buffers=True, **kwargs):
    """
    Federated averaging of model parameters.
    """
    print("\n[DEBUG] Entered fedavg()")
    
    # Print types of inputs
    print(f"[DEBUG] server_model_state_dict type: {type(server_model_state_dict)}")
    print(f"[DEBUG] client_updates_dict type: {type(client_updates_dict)}")
    
    # Print a few keys of the server model
    if isinstance(server_model_state_dict, dict):
        print(f"[DEBUG] server_model_state_dict keys (sample): {list(server_model_state_dict.keys())[:3]}")
    
    # Print client_updates_dict summary
    if isinstance(client_updates_dict, dict):
        print(f"[DEBUG] Number of clients: {len(client_updates_dict)}")
        first_client = next(iter(client_updates_dict))
        print(f"[DEBUG] First client ID: {first_client}")
        print(f"[DEBUG] First client update type: {type(client_updates_dict[first_client])}")
        print(f"[DEBUG] First client update keys (sample): {list(client_updates_dict[first_client].keys())[:3]}")
    else:
        print("[ERROR] client_updates_dict is not a dict!")

    # Turn updates into a list
    model_list = list(client_updates_dict.values())
    
    # Check what's inside model_list
    if not model_list:
        raise ValueError("[ERROR] model_list is empty!")

    if not isinstance(model_list[0], dict):
        raise TypeError(f"[ERROR] Expected model_list[0] to be a dict, got {type(model_list[0])}")

    # Start aggregation
    avg_model = {}

    for key in model_list[0].keys():
        if 'num_batches_tracked' in key:
            avg_model[key] = model_list[0][key].clone()
        elif any(buf in key for buf in ['running_mean', 'running_var']):
            if average_bn_buffers:
                avg_model[key] = torch.stack([m[key] for m in model_list]).mean(0)
        else:
            avg_model[key] = torch.stack([m[key] for m in model_list]).mean(0)

    print("[DEBUG] Aggregation complete. Returning averaged model.")
    return avg_model



def trimmed_mean(server_model_state_dict, client_updates_dict, m, **kwargs):
    new_weights = dict()
    client_ids = list(client_updates_dict.keys())

    with torch.no_grad():
        for layer_name, param in client_updates_dict[client_ids[0]].items():
            size = param.size()
            layer_weights = []
            for client_id in client_ids:
                layer_weight = client_updates_dict[client_id][layer_name]
                layer_weights.append(torch.flatten(layer_weight))

            t = torch.stack(layer_weights)
            meds = torch.median(t, 0).values
            dist = -(t - meds.repeat(t.size()[0], 1))**2
            m_closet_idx = torch.topk(dist, m, dim=0).indices
            # print("m_closet_idx", m_closet_idx.size())
            y_axis = torch.arange(0, t.size()[1]).unsqueeze(0).repeat(m,1)
            # print("y_axis", y_axis.size())
            m_closet = t[m_closet_idx.flatten(start_dim=0), y_axis.flatten(start_dim=0).long(),...].view(m,-1)
            mean_m_closet = torch.mean(m_closet.float(), 0)
            new_weights[layer_name] = mean_m_closet.view(size)
    # print("new_weights.keys", new_weights.keys())
    return new_weights


def get_layer_weight(model, name_list):
    splitted = name_list.split('.')
    result = model
    for p in splitted:
        result = getattr(result, p)
    return result


