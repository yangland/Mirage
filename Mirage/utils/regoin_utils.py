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


def build_region_constraints(
    stats,
    l2_scale_min=1.5,
    l2_scale_max=10.0,
    cos_scale_min=0.75
):
    """
    Build constraint dicts for regions R1–R4 using L2 and cosine constraints.

    Args:
        stats: Dictionary with benign model statistics (from compute_benign_statistics)
        l2_scale_min: Minimum scaling factor for L2 norm (e.g., 1.5)
        l2_scale_max: Maximum scaling factor for L2 norm (e.g., 10)
        cos_scale_min: Scaling factor for cosine threshold (e.g., 0.75)

    Returns:
        Dictionary mapping region_id to constraint dict
    """
    constraints = {}

    # === Shared values ===
    avg_benign_model = stats["avg_benign_model"]
    avg_L2_norm = stats["avg_L2_norm"]
    avg_pairwise_cosine_distance = stats["avg_update_cos_d"]
    avg_update_cos_d_to_benign = stats["avg_update_cos_d_to_benign"]

    r_min = avg_L2_norm * l2_scale_min
    r_max = avg_L2_norm * l2_scale_max
    cosine_threshold = avg_update_cos_d_to_benign * cos_scale_min

    # === Region R1: Small norm, aligned ===
    constraints[1] = {
        "region_id": 1,
        "avg_benign_weight": avg_benign_model,
        "l2_radius": r_min,
        "update_cone_mode": 1,
        "cosine_threshold": cosine_threshold,
    }

    # === Region R2: Large norm, aligned ===
    constraints[2] = {
        "region_id": 2,
        "avg_benign_weight": avg_benign_model,
        "l2_radius": r_max,
        "update_cone_mode": 1,
        "cosine_threshold": cosine_threshold,
    }

    # === Region R3: Small norm, opposite direction ===
    constraints[3] = {
        "region_id": 3,
        "avg_benign_weight": avg_benign_model,
        "l2_radius": r_min,
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



def search_k_percent_to_fix_geometry_old(delta_theta, delta_b, update_cone_mode, cosine_threshold):
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

        valid, _ = check_cos_constraint(
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


def search_k_percent_to_fix_geometry(delta_theta, delta_b, update_cone_mode, cosine_threshold):
    """
    Binary search for the minimum k% of elements to replace with delta_b (or -delta_b)
    to satisfy the geometric (cosine) constraint.

    Returns:
        - fixed_delta_theta: Modified update
        - final_k_percent: Minimum percent that satisfies the constraint (or None)
    """
    delta_original = delta_theta.clone()
    low, high = 0, 100
    best_k = None
    best_delta = delta_original
    found_valid = False

    while low <= high:
        mid = (low + high) // 2
        k = mid / 100.0

        modified_delta = replace_k_percent_with_target(
            delta_theta=delta_original,
            delta_b=delta_b,
            k_percent=k,
            update_cone_mode=update_cone_mode
        )

        valid, _ = check_cos_constraint(
            delta_theta=modified_delta,
            delta_b=delta_b,
            update_cone_mode=update_cone_mode,
            cosine_threshold=cosine_threshold
        )

        if valid:
            found_valid = True
            best_k = mid
            best_delta = modified_delta
            high = mid - 1
        else:
            low = mid + 1

    return best_delta, best_k



def replace_k_percent_with_target(delta_theta, delta_b, k_percent, update_cone_mode):
    """
    Replace k% of entries that hurt alignment the most.
    For align mode (+1): smallest contributions to dot(Δ, Δb) get replaced.
    For oppose mode (-1): smallest contributions to dot(Δ, -Δb) get replaced.
    """
    target = delta_b if update_cone_mode == 1 else -delta_b
    contrib = delta_theta * target  # per-dim contribution to alignment

    numel = delta_theta.numel()
    k = int(round(k_percent * numel))
    if k <= 0:
        return delta_theta

    # indices that most hurt alignment (smallest contrib)
    idx = torch.topk(contrib, k, largest=False).indices
    modified = delta_theta.clone()
    modified[idx] = target[idx]
    return modified




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


def check_cos_constraint(delta_theta, delta_b, update_cone_mode, cosine_threshold):
    """
    Check if the cosine distance between delta_theta and delta_b
    satisfies the geometric constraint.

    Returns:
        is_valid (bool): Whether the constraint is satisfied
        cos_dist (float): The actual cosine distance
        cosine_threshold (float): The constraint threshold
    """
    if delta_b is None or delta_theta is None:
        return True, None, cosine_threshold  # Avoid crashing

    norm_b = torch.norm(delta_b)
    norm_theta = torch.norm(delta_theta)

    if norm_b == 0 or norm_theta == 0:
        return True, None, cosine_threshold  # Avoid division by zero

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(
        delta_theta.unsqueeze(0), delta_b.unsqueeze(0), eps=1e-8
    ).item()
    cos_dist = 1.0 - cos_sim  # Convert to cosine distance

    if update_cone_mode == 1:
        is_valid = cos_dist < cosine_threshold  # Require alignment
    elif update_cone_mode == -1:
        is_valid = cos_dist > cosine_threshold  # Require dissimilarity
    else:
        raise ValueError(f"[ERROR] Invalid update_cone_mode: {update_cone_mode}")

    return is_valid, cos_dist



from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np

def compute_benign_statistics(benign_models, server_model):
    """
    Computes:
    - avg_benign_model: θ̄_b
    - avg_L2_dist: average pairwise L2 distance between benign models
    - avg_L2_norm: average norm of model updates from server
    - avg_update_cos_d: average cosine angle between benign updates
    - avg_weight_cos_d: average cosine angle between benign model weights
    - avg_update_cos_d_to_benign: average cosine distance of each update to the benign-average update  # NEW
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

    # --- NEW: benign-average update vector Δθ̄ = θ̄_b - θ_server ---
    delta_avg = flatten_model(avg_model) - flatten_model(server_model)  # NEW

    # === Compute statistics ===
    update_dists = []
    weight_dists = []
    l2_dists = []
    l2_norms = []
    update_to_benign_dists = []  # NEW

    for i in range(M):
        model_i = benign_models[i]

        # L2 norm from server model (‖Δθ_i‖)
        delta_i = flatten_model(model_i) - flatten_model(server_model)
        l2_norms.append(torch.norm(delta_i, p=2).item())

        # --- NEW: cosine distance between Δθ_i and Δθ̄ ---
        cos_sim_to_avg = F.cosine_similarity(delta_i.unsqueeze(0), delta_avg.unsqueeze(0)).item()  # NEW
        update_to_benign_dists.append(1 - cos_sim_to_avg)  # NEW

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
        "avg_update_cos_d_to_benign": np.mean(update_to_benign_dists),  # NEW
    }

    return stat



@torch.no_grad()
def project_update_inplace_(model: torch.nn.Module,
                            center_model: torch.nn.Module,
                            radius: float) -> None:
    """Project θ onto the L2 ball around center with radius, in-place."""
    theta  = flatten_model(model)
    center = flatten_model(center_model)

    diff = theta - center
    norm = diff.norm(p=2)

    if norm > radius:
        theta_proj = center + diff * (radius / (norm + 1e-12))
        _assign_flat_params_(model, theta_proj)  # in-place copy into existing params


@torch.no_grad()
def scale_update_to_l2_boundary_inplace_(model: torch.nn.Module,
                                         center_model: torch.nn.Module,
                                         radius: float) -> None:
    """Scale Δ = θ - center to have ‖Δ‖2 = radius, in-place."""
    theta  = flatten_model(model)
    center = flatten_model(center_model)
    delta  = theta - center
    norm   = delta.norm()

    if norm < 1e-12:
        print("[WARNING] Update norm too small; skip scaling.")
        return

    theta_scaled = center + delta * (radius / (norm + 1e-12))
    _assign_flat_params_(model, theta_scaled)  # in-place copy


@torch.no_grad()
def _assign_flat_params_(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Copy a flat vector into model parameters without re-creating tensors."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(flat[offset:offset+n].view_as(p))
        offset += n




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



def compute_geo_loss(delta_theta, delta_b, update_cone_mode, threshold=1e-6):
    """
    Computes geometric alignment loss between delta_theta and delta_b.
    Skips loss if delta_theta or delta_b is too small.
    """
    norm_theta = delta_theta.norm()
    norm_b = delta_b.norm()

    if norm_theta.item() < threshold or norm_b.item() < threshold:
        # Return a zero loss that supports autograd
        return (delta_theta * 0.0).sum() # graph-connected zero

    # Cosine similarity ∈ [-1, 1], clamp to avoid numerical issues
    cos_sim = F.cosine_similarity(delta_theta.unsqueeze(0), delta_b.unsqueeze(0), eps=1e-8)
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    if update_cone_mode == 1:
        return 1.0 - cos_sim  # alignment
    elif update_cone_mode == -1:
        return 1.0 + cos_sim  # opposition
    else:
        raise ValueError(f"Invalid update_cone_mode: {update_cone_mode}")


def scale_model_update_to_l2_boundary(
    cache_model: torch.nn.Module,
    global_model: torch.nn.Module,
    l2_radius: float
) -> torch.nn.Module:
    """
    Scales the update from global_model to cache_model so that its L2 norm equals l2_radius.
    """
    delta = flatten_model(cache_model) - flatten_model(global_model)
    norm = delta.norm()

    if norm < 1e-12:
        print("[WARNING] Update norm too small; skipping scaling.")
        return cache_model

    scaled_delta = delta / norm * l2_radius
    new_flat_params = flatten_model(global_model) + scaled_delta
    updated_model= unflatten_model(new_flat_params, cache_model)

    return updated_model


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
