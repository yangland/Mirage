# utils/alpha_estimation.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Any

def _sd_update_to_vec(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten one client's update dict -> 1D vector on CPU (ignores BN counters)."""
    vecs = []
    for k, v in update.items():
        if k.endswith("num_batches_tracked"):
            continue
        vecs.append(v.reshape(-1).detach().cpu().float())
    return torch.cat(vecs) if vecs else torch.zeros(1)

def _avg_updates(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average a list of update dicts (same keys)."""
    if not updates:
        return {}
    out = {}
    keys = updates[0].keys()
    for k in keys:
        if k.endswith("num_batches_tracked"):
            out[k] = updates[0][k].clone()
        else:
            out[k] = torch.stack([u[k] for u in updates], dim=0).mean(dim=0)
    return out

def estimate_alpha_update_mixture(
    global_delta_vec: torch.Tensor,
    mal_delta_vec: torch.Tensor,
    ben_delta_vec: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[float, dict]:
    """
    Solve g ≈ a*m + b*b via least squares; return a (clamped to [0,1]).
    Fallback: 1D projection onto (m-b) if matrix is ill-conditioned.
    """
    # Stack columns [m b]
    M = torch.stack([mal_delta_vec, ben_delta_vec], dim=1)  # [D,2]
    ATA = (M.T @ M).cpu().numpy()  # 2x2
    ATg = (M.T @ global_delta_vec).cpu().numpy()  # 2

    # Ridge for stability
    ATA_r = ATA + eps * np.eye(2, dtype=ATA.dtype)
    try:
        coeff = np.linalg.solve(ATA_r, ATg)
        a, b = float(coeff[0]), float(coeff[1])
        mode = "lsq"
    except np.linalg.LinAlgError:
        # fallback: project onto (m - b)
        u = (mal_delta_vec - ben_delta_vec)
        denom = float((u @ u).item())
        if denom < eps:
            return 0.0, {"mode": "fallback_colinear", "a_raw": 0.0, "b_raw": None}
        a = float(((global_delta_vec - ben_delta_vec) @ u).item() / denom)
        b = None
        mode = "proj_mb"

    alpha = max(0.0, min(1.0, a))
    debug = {"mode": mode, "a_raw": a, "b_raw": b}
    return alpha, debug

def make_round_alpha_rows(
    *,
    iteration: int,
    prev_global_sd: Dict[str, torch.Tensor],
    new_global_sd: Dict[str, torch.Tensor],
    updates_by_client: Dict[int, Dict[str, torch.Tensor]],
    selected_clients: List[int],
    malicious_clients: List[int],
    client_region_mapping: Dict[int, int],
    region_constraints_dict: Dict[int, dict],
) -> List[dict]:
    """
    Build alpha rows for all regions attacked this round.
    Uses update-space LSQ to estimate inclusion weight for each attacked region.
    """
    # Compute global round delta (new - prev)
    g_delta = {}
    for k, v in new_global_sd.items():
        if k.endswith("num_batches_tracked"):  # keep shape consistency; value ignored in vec
            g_delta[k] = v.detach() - prev_global_sd[k].detach()
        else:
            g_delta[k] = v.detach() - prev_global_sd[k].detach()
    g_vec = _sd_update_to_vec(g_delta)

    # Partition client updates
    benign_ids = [cid for cid in selected_clients if cid not in malicious_clients]
    rows = []

    # Average benign updates in this round (what actually contributed)
    ben_updates = [updates_by_client[cid] for cid in benign_ids if cid in updates_by_client]
    ben_avg = _avg_updates(ben_updates)
    if not ben_avg:
        # Degenerate (shouldn't happen in your setup)
        ben_avg = {k: torch.zeros_like(v) for k, v in g_delta.items()}
    b_vec = _sd_update_to_vec(ben_avg)

    # Group malicious by region
    by_region = {}
    for mid in malicious_clients:
        rid = client_region_mapping.get(mid, None)
        if rid is None:
            continue
        by_region.setdefault(rid, []).append(mid)

    P = len(selected_clients)
    for rid, mids in by_region.items():
        mal_updates = [updates_by_client[cid] for cid in mids if cid in updates_by_client]
        if not mal_updates:
            continue
        mal_avg = _avg_updates(mal_updates)
        m_vec = _sd_update_to_vec(mal_avg)

        alpha, dbg = estimate_alpha_update_mixture(g_vec, m_vec, b_vec)

        c = region_constraints_dict.get(rid, {})
        rows.append({
            "iteration": iteration,
            "region_id": int(rid),
            "k_mal": int(len(mids)),
            "P_total": int(P),
            "alpha_update": float(alpha),
            "est_mode": dbg.get("mode", "lsq"),
            "a_raw": dbg.get("a_raw", None),
            "b_raw": dbg.get("b_raw", None),
            # geometry snapshot for convenience (duplicated in geometry csv too)
            "l2_radius": float(c.get("l2_radius", float("nan"))) if c else float("nan"),
            "cosine_threshold": float(c.get("cosine_threshold", float("nan"))) if c else float("nan"),
            "update_cone_mode": int(c.get("update_cone_mode", 0)) if c else 0,
        })
    return rows
