import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from copy import deepcopy

def flatten_model(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def unflatten_model(flat_tensor, model_template):
    """
    Reconstructs a model from a flat tensor using the structure of model_template.
    
    Args:
        flat_tensor (torch.Tensor): 1D tensor containing all parameters.
        model_template (torch.nn.Module): A model with the desired architecture.

    Returns:
        model: A new model with parameters from flat_tensor.
    """
    new_model = deepcopy(model_template)
    current_index = 0

    for param in new_model.parameters():
        numel = param.numel()
        param_shape = param.shape
        param.data.copy_(
            flat_tensor[current_index:current_index + numel].view(param_shape)
        )
        current_index += numel

    return new_model

def compute_model_distance(a, b, norm=2):
    return torch.norm(flatten_model(a) - flatten_model(b), p=norm).item()

def cosine_distance(a, b):
    a_flat = flatten_model(a)
    b_flat = flatten_model(b)
    return 1 - F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

# Bit	Meaning
# 0	Inside L2-ball
# 1	Inside Update Cone
# 2	Inside Weight Cone

def get_region_id(in_l2, in_update_cone, in_weight_cone):
    return (in_weight_cone << 2) | (in_update_cone << 1) | in_l2


def build_region_constraints(stats):
    """
    Given benign statistics, return constraint dicts for all 8 regions.
    Each region ID maps to a dict of constraints.
    """
    constraints = {}
    for region_id in range(8):
        in_l2 = region_id & 1
        in_update = (region_id >> 1) & 1
        in_weight = (region_id >> 2) & 1

        region_constraints = {
            "apply_l2": bool(in_l2),
            "apply_update_cone": bool(in_update),
            "apply_weight_cone": bool(in_weight),
            "center": stats["avg_benign_model"],  # common in all
            "l2_radius": stats["l2_radius"] if in_l2 else None,
            "update_cone_angle": stats["update_cone_angle"] if in_update else None,
            "weight_cone_angle": stats["weight_cone_angle"] if in_weight else None,
        }
        constraints[region_id] = region_constraints
    return constraints


def compute_benign_statistics(benign_models, server_model):
    """
    Computes:
    - avg_benign_model
    - avg L2 distance
    - avg update cosine distance
    - avg weight cosine distance
    """
    M = len(benign_models)
    assert M > 1, "Need at least 2 clients for statistics"

    # Compute average benign model (θ̄_b)
    avg_benign_state = defaultdict(lambda: 0)
    for model in benign_models:
        for name, param in model.state_dict().items():
            avg_benign_state[name] += param.clone()
    for name in avg_benign_state:
        avg_benign_state[name] /= M

    # Create dummy model with avg weights
    avg_model = deepcopy(benign_models[0])
    avg_model.load_state_dict(avg_benign_state)

    # Compute distances
    update_dists = []
    weight_dists = []
    l2_dists = []

    for i in range(M):
        for j in range(i + 1, M):
            model_i, model_j = benign_models[i], benign_models[j]

            # L2 distance
            l2_dists.append(compute_model_distance(model_i, model_j))

            # Update direction
            delta_i = flatten_model(model_i) - flatten_model(server_model)
            delta_j = flatten_model(model_j) - flatten_model(server_model)
            update_dists.append(1 - F.cosine_similarity(delta_i.unsqueeze(0), delta_j.unsqueeze(0)).item())

            # Weight direction
            weight_dists.append(cosine_distance(model_i, model_j))

    stat = {
        "avg_benign_model": avg_model,
        "l2_radius": np.mean(l2_dists),
        "update_cone_angle": np.mean(update_dists),
        "weight_cone_angle": np.mean(weight_dists),
    }

    return stat

def project_model_into_region(model, server_model, region_constraints):
    """
    Approximate projection of model into constrained region.
    You can apply clipping if needed in PGD loop.
    """
    projected_model = deepcopy(model)

    theta = flatten_model(model)
    center = flatten_model(region_constraints["center"])

    # L2 projection
    if region_constraints["apply_l2"]:
        diff = theta - center
        norm = torch.norm(diff, p=2)
        if norm > region_constraints["l2_radius"]:
            theta = center + diff / norm * region_constraints["l2_radius"]

    # NOTE: Update cone and weight cone projections can be approximated
    # with gradient filtering or angular clipping — or used as constraints
    # in PGD step instead of projection.

    return unflatten_model(theta, model)


def is_within_l2_ball(model, benign_model, l2_radius):
    return compute_model_distance(model, benign_model) <= l2_radius

def is_within_update_cone(model, server_model, avg_benign_model, update_cone_angle):
    theta_t = flatten_model(server_model)
    theta = flatten_model(model)
    delta_b = flatten_model(avg_benign_model) - theta_t
    delta = theta - theta_t
    cos_dist = 1 - F.cosine_similarity(delta.unsqueeze(0), delta_b.unsqueeze(0)).item()
    return cos_dist <= update_cone_angle

def is_within_weight_cone(model, avg_benign_model, weight_cone_angle):
    theta = flatten_model(model)
    theta_bar = flatten_model(avg_benign_model)
    delta = theta - theta_bar
    cos_dist = 1 - F.cosine_similarity(delta.unsqueeze(0), theta_bar.unsqueeze(0)).item()
    return cos_dist <= weight_cone_angle
