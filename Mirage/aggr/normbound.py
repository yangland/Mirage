import torch
from numpy import median

def normbound_aggr(server_sd, clients_grad_dict):
    server_weight = flat_dict(server_sd)
    updates = []
    for client_id, client_model_sd in clients_grad_dict.items():
        new_model_i = []
        for name in server_sd:
            new_model_i.append(torch.flatten(client_model_sd[name].detach()))
        new_model_i = torch.cat(new_model_i)
        assert new_model_i.size() == server_weight.size(), "Mismatch in model size"
        updates_i = new_model_i 
        updates.append(updates_i)

    norm_list = [update.norm().unsqueeze(0) for update in updates]
    norm_tensor = torch.cat(norm_list, dim=0)
    median_value = norm_tensor.median().item()

    clipped_updates = [
        update * min(1.0, (median_value + 1e-8) / (update.norm() + 1e-8)) if update.norm() > 0 else update
        for update in updates
    ]

    clipped_update = torch.mean(torch.stack(clipped_updates), dim=0)
    reference_sd = next(iter(clients_grad_dict.values()))
    clipped_grad = restore_dict_grad_flat(clipped_update, reference_sd)
    return clipped_grad


def flat_dict(grad_dict, layer_list=None):
    user_flatten_grad = []
    keys = grad_dict.keys() if layer_list is None else layer_list
    for name in keys:
        user_flatten_grad.append(torch.flatten(grad_dict[name].detach()))
    return torch.cat(user_flatten_grad)


def restore_dict_grad_flat(flat_grad, model_dict):
    restored_w = {}
    start = 0
    for name, param in model_dict.items():
        num_elements = param.numel()
        restored_w[name] = flat_grad[start:start + num_elements].view(param.shape)
        start += num_elements
    return restored_w
