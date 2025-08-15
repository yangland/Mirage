# utils/alpha_estimation.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def _l2(v: torch.Tensor) -> float:
    return float(v.norm().item())

def _cos_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    an = a.norm(); bn = b.norm()
    if an.item() < eps or bn.item() < eps:
        return float("nan")
    cos_sim = float((a @ b) / (an * bn))
    return 1.0 - cos_sim

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
    sim_benign_avg_update: Optional[Dict[str, torch.Tensor]] = None,  # provided from broadcast_upload
    real_benign_avg_update: Optional[Dict[str, torch.Tensor]] = None, # provided from broadcast_upload
) -> List[dict]:
    """
    For each attacked region:
      - Estimate alpha using *simulated* benign reference (malicious-view).
      - Also estimate alpha using *real* benign reference (server-view).
      - Log both sets with and without `_real` suffix.
      - Include diagnostics of sim-vs-real benign discrepancy.
    """
    # Global round delta
    g_delta = {k: (new_global_sd[k].detach() - prev_global_sd[k].detach()) for k in new_global_sd.keys()}
    g_vec = _sd_update_to_vec(g_delta)

    # Benign refs → vectors
    b_sim_vec  = _sd_update_to_vec(sim_benign_avg_update)  if sim_benign_avg_update  is not None else None
    b_real_vec = _sd_update_to_vec(real_benign_avg_update) if real_benign_avg_update is not None else None

    # Sim-vs-real diagnostics (round-level)
    b_real_l2 = _l2(b_real_vec) if b_real_vec is not None else float("nan")
    b_sim_l2  = _l2(b_sim_vec)  if b_sim_vec  is not None else float("nan")
    b_cos_d   = _cos_dist(b_real_vec, b_sim_vec) if (b_real_vec is not None and b_sim_vec is not None) else float("nan")
    b_l2_pct_diff = (abs(b_real_l2 - b_sim_l2) / max(b_real_l2, 1e-12)) if (b_real_vec is not None and b_sim_vec is not None) else float("nan")

    # Group malicious virtual clients by region
    by_region: Dict[int, List[int]] = {}
    for mid in malicious_clients:
        rid = client_region_mapping.get(mid)
        if rid is None:
            continue
        by_region.setdefault(rid, []).append(mid)

    P = len(selected_clients)
    rows: List[dict] = []

    for rid, mids in by_region.items():
        # average malicious update for this region
        mal_updates = [updates_by_client[cid] for cid in mids if cid in updates_by_client]
        if not mal_updates:
            continue
        m_vec = _sd_update_to_vec(_avg_updates(mal_updates))

        c = region_constraints_dict.get(rid, {})
        row = {
            "iteration": int(iteration),
            "region_id": int(rid),
            "k_mal": int(len(mids)),
            "P_total": int(P),
            # benign diagnostics (sim vs real) — constant across refs
            "b_real_l2": float(b_real_l2),
            "b_sim_l2":  float(b_sim_l2),
            "b_cos_dist": float(b_cos_d),
            "b_l2_pct_diff": float(b_l2_pct_diff),
            # geometry snapshot
            "l2_radius": float(c.get("l2_radius", float("nan"))) if c else float("nan"),
            "cosine_threshold": float(c.get("cosine_threshold", float("nan"))) if c else float("nan"),
            "update_cone_mode": int(c.get("update_cone_mode", 0)) if c else 0,
        }

        # Compute alphas for both references that are available
        ref_list = [
            ("sim",  b_sim_vec,  ""),       # no suffix for simulated (malicious-view)
            ("real", b_real_vec, "_real"),  # `_real` suffix for server-view
        ]
        for ref_name, b_vec, suffix in ref_list:
            if b_vec is None:
                # Keep the keys present with NaNs so CSV schema is stable
                row[f"alpha_update{suffix}"] = float("nan")
                row[f"est_mode{suffix}"]     = None
                row[f"a_raw{suffix}"]        = None
                row[f"b_raw{suffix}"]        = None
                continue

            alpha, dbg = estimate_alpha_update_mixture(g_vec, m_vec, b_vec)
            row[f"alpha_update{suffix}"] = float(alpha)
            row[f"est_mode{suffix}"]     = dbg.get("mode", "lsq")
            row[f"a_raw{suffix}"]        = dbg.get("a_raw", None)
            row[f"b_raw{suffix}"]        = dbg.get("b_raw", None)

        # Optional: direct alpha discrepancy metrics (malicious-view vs server-view)
        a_sim  = row.get("alpha_update")
        a_real = row.get("alpha_update_real")
        if (a_sim is not None) and (a_real is not None) and all(map(lambda x: isinstance(x, float), [a_sim, a_real])) \
           and (not (torch.isnan(torch.tensor(a_sim)) or torch.isnan(torch.tensor(a_real)))):
            row["alpha_gap_abs"] = abs(a_sim - a_real)
            denom = max(abs(a_real), 1e-12)  # relative to server-view alpha
            row["alpha_gap_rel"] = abs(a_sim - a_real) / denom
        else:
            row["alpha_gap_abs"] = float("nan")
            row["alpha_gap_rel"] = float("nan")

        rows.append(row)

    return rows
