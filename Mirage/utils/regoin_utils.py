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


def build_region_constraints(stats, l2_radius_scale=5.0):
    """
    Build constraint dicts for regions R1–R4 using L2 and cosine constraints.
    
    Args:
        stats: Dictionary with benign model statistics (from compute_benign_statistics)
        l2_radius_scale: Scaling factor for r_max (e.g., 5 or 10)

    Returns:
        Dictionary mapping region_id to constraint dict
    """
    constraints = {}

    # === Shared values ===
    avg_benign_model = stats["avg_benign_model"]
    avg_L2_norm = stats["avg_L2_norm"]
    avg_cosine_distance = stats["avg_update_cos_d"]
    r_max = avg_L2_norm * l2_radius_scale

    # === Region R1: Small norm, aligned (update_cone_mode = 1) ===
    constraints[1] = {
        "region_id": 1,
        "avg_benign_weight": avg_benign_model,
        "l2_radius": avg_L2_norm,
        "update_cone_mode": 1,
        "cosine_threshold": avg_cosine_distance,
    }

    # === Region R2: Large norm, aligned ===
    constraints[2] = {
        "region_id": 2,
        "avg_benign_weight": avg_benign_model,
        "l2_radius": r_max,
        "update_cone_mode": 1,
        "cosine_threshold": avg_cosine_distance,
    }

    # === Region R3: Small norm, opposite direction ===
    constraints[3] = {
        "region_id": 3,
        "avg_benign_weight": avg_benign_model,
        "l2_radius": avg_L2_norm,
        "update_cone_mode": -1,
        "cosine_threshold": 1.0,
    }

    # === Region R4: Large norm, opposite direction ===
    constraints[4] = {
        "region_id": 4,
        "avg_benign_weight": avg_benign_model,
        "l2_radius": r_max,
        "update_cone_mode": -1,
        "cosine_threshold": 1.0,
    }

    return constraints


def search_k_percent_to_fix_geometry(delta_theta, delta_b, update_cone_mode, cosine_threshold):
    """
    Binary search for the minimum k% to flip that makes the geometric constraint valid.
    Returns:
        - fixed_delta_theta, final_k_percent
    """
    delta_original = delta_theta.clone()
    low, high = 0, 100
    best_delta = delta_original
    found_valid = False

    while low <= high:
        mid = (low + high) // 2
        k = mid / 100.0
        flipped_delta = flip_bottom_k_percent(delta_original, k)

        valid = check_geometric_constraint(
            delta_theta=flipped_delta,
            delta_b=delta_b,
            update_cone_mode=update_cone_mode,
            cosine_threshold=cosine_threshold
        )

        if valid:
            found_valid = True
            best_delta = flipped_delta
            high = mid - 1  # Try smaller k
        else:
            low = mid + 1  # Try larger k

    return best_delta, (low if found_valid else None)


def flip_bottom_k_percent(delta_theta, k_percent):
    """
    Flip the sign of the bottom-k% smallest-magnitude elements in delta_theta.
    """
    delta = delta_theta.clone()
    flat = delta.abs()
    k = int(len(flat) * k_percent)
    if k == 0:
        return delta

    threshold = torch.topk(flat, k, largest=False).values[-1]
    flip_mask = flat <= threshold
    delta[flip_mask] *= -1
    return delta


def check_geometric_constraint(delta_theta, delta_b, update_cone_mode, cosine_threshold):
    """
    Check if the cosine similarity between delta_theta and delta_b
    satisfies the geometric constraint.

    - For update_cone_mode == -1 → require: cos_sim > threshold
    - For update_cone_mode == +1 → require: cos_sim < threshold
    """
    if delta_b is None or torch.norm(delta_b) == 0 or torch.norm(delta_theta) == 0:
        return True  # Avoid false negatives

    cos_sim = F.cosine_similarity(delta_theta.unsqueeze(0), delta_b.unsqueeze(0), eps=1e-8).item()

    if update_cone_mode == -1:
        return cos_sim > cosine_threshold
    elif update_cone_mode == 1:
        return cos_sim < cosine_threshold
    else:
        raise ValueError(f"[ERROR] Invalid update_cone_mode: {update_cone_mode}")


def compute_benign_statistics(benign_models, server_model):
    """
    Computes:
    - avg_benign_model: θ̄_b
    - avg_L2_dist: average pairwise L2 distance between benign models
    - avg_L2_norm: average norm of model updates from server
    - avg_update_cos_d: average cosine angle between benign updates
    - avg_weight_cos_d: average cosine angle between benign model weights
    """
    M = len(benign_models)
    assert M > 1, "Need at least 2 clients for statistics"

    # === Compute average benign model (θ̄_b) ===
    avg_benign_state = {}
    for name in benign_models[0].state_dict().keys():
        avg_tensor = benign_models[0].state_dict()[name].float().clone()
        for model in benign_models[1:]:
            avg_tensor += model.state_dict()[name].float()
        avg_tensor /= M
        avg_benign_state[name] = avg_tensor.to(dtype=benign_models[0].state_dict()[name].dtype)

    # Create model with avg weights
    avg_model = deepcopy(benign_models[0])
    avg_model.load_state_dict(avg_benign_state)

    # === Compute statistics ===
    update_dists = []
    weight_dists = []
    l2_dists = []
    l2_norms = []

    for i in range(M):
        model_i = benign_models[i]

        # L2 norm from server model (‖Δθ_i‖)
        delta_i = flatten_model(model_i) - flatten_model(server_model)
        l2_norms.append(torch.norm(delta_i, p=2).item())

        for j in range(i + 1, M):
            model_j = benign_models[j]

            # Pairwise L2 distance (‖θ_i - θ_j‖)
            l2_dists.append(compute_model_distance(model_i, model_j))

            # Update cone: cos angle between Δθ_i and Δθ_j
            delta_j = flatten_model(model_j) - flatten_model(server_model)
            cos_sim = F.cosine_similarity(delta_i.unsqueeze(0), delta_j.unsqueeze(0)).item()
            update_dists.append(1 - cos_sim)

            # Weight cone: cos angle between θ_i and θ_j
            weight_dists.append(cosine_distance(model_i, model_j))

    stat = {
        "avg_benign_model": avg_model,
        "avg_L2_dist": np.mean(l2_dists),
        "avg_L2_norm": np.mean(l2_norms),
        "avg_update_cos_d": np.mean(update_dists),
        "avg_weight_cos_d": np.mean(weight_dists),
    }

    return stat


def project_model_into_region(model, center_model, radius):
    """
    Projects the model into an L2 ball around the given center model.
    """
    theta = flatten_model(model)
    center = flatten_model(center_model)

    diff = theta - center
    norm = torch.norm(diff, p=2)

    if norm > radius:
        theta = center + diff / norm * radius

    return unflatten_model(theta, model)


# def compute_geo_loss(delta_theta, theta, global_model, avg_benign_model, region_id):
#     if region_id == 1:  # 𝓧₁: No geo loss
#         return 0.0

#     elif region_id == 2:  # 𝓧₂: Update cone
#         delta_b = flatten_model(avg_benign_model) - flatten_model(global_model)
#         cos_sim = F.cosine_similarity(delta_theta.unsqueeze(0), 
#                                       delta_b.unsqueeze(0), 
#                                       eps=1e-8
#                                       ).item()
#         return -cos_sim

#     elif region_id == 4:  # 𝓧₃: Weight cone
#         theta_b = flatten_model(avg_benign_model)
#         theta_new = flatten_model(theta)
#         delta = theta_new - theta_b
        
#         if torch.norm(theta_b) == 0 or torch.norm(delta) == 0:
#             print("[WARNING] Zero norm in geo loss for region 4")
#             return 0.0
        
#         cos_sim = F.cosine_similarity(delta.unsqueeze(0),
#                                       theta_b.unsqueeze(0), 
#                                       eps=1e-8
#                                     ).item()
#         return -cos_sim

#     else:  # 𝓧₄ or others
#         return 0.0


def apply_delta_to_model(base_model, delta_theta):
    """
    Applies the delta_theta update to base_model and returns a new model instance.

    Args:
        base_model (torch.nn.Module): The original (global) model.
        delta_theta (torch.Tensor): Flattened parameter update to apply.

    Returns:
        new_model (torch.nn.Module): Model after applying the update.
    """
    base_flat = flatten_model(base_model)
    updated_flat = base_flat + delta_theta
    new_model = unflatten_model(updated_flat, base_model)
    return new_model



def compute_geo_loss(delta_theta, delta_b, update_cone_mode):
    """
    Computes geometric loss based on region type:
    - update_cone_mode = 1: encourage alignment (1 - cos_sim)
    - update_cone_mode = -1: encourage opposition (cos_sim)
    """
    if delta_b is None or torch.norm(delta_b) == 0 or torch.norm(delta_theta) == 0:
        return 0.0

    cos_sim = F.cosine_similarity(delta_theta.unsqueeze(0), delta_b.unsqueeze(0), eps=1e-8).item()

    if update_cone_mode == 1:
        return 1 - cos_sim  # alignment
    elif update_cone_mode == -1:
        return cos_sim      # opposition
    else:
        raise ValueError(f"Invalid update_cone_mode: {update_cone_mode}")


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
