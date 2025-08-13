# utils/agg_pref_summary.py
import json, numpy as np, csv
from collections import defaultdict

def _load_alpha(alpha_csv_path):
    rows = []
    with open(alpha_csv_path, "r") as f:
        r = csv.DictReader(f)
        for x in r:
            try:
                rows.append({
                    "iteration": int(x["iteration"]),
                    "region_id": int(x["region_id"]),
                    "alpha_update": float(x["alpha_update"]),
                    "l2_radius": float(x.get("l2_radius","nan")),
                    "update_cone_mode": int(x.get("update_cone_mode","0")),
                })
            except:
                pass
    return rows

def _normalize01(arr):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any(): return np.zeros_like(arr)
    lo, hi = arr[finite].min(), arr[finite].max()
    if hi <= lo: return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return out

def summarize_preferences(alpha_csv_path: str, out_json_path: str):
    rows = _load_alpha(alpha_csv_path)
    if not rows:
        with open(out_json_path, "w") as f:
            json.dump({"error":"no alpha rows"}, f, indent=2)
        return

    alphas = np.array([r["alpha_update"] for r in rows], dtype=float)
    norms  = _normalize01([r["l2_radius"] for r in rows])
    dirs   = np.array([1.0 if r["update_cone_mode"]==1 else 0.0 for r in rows], dtype=float)

    # Design matrix with ridge
    X = np.column_stack([np.ones_like(alphas), norms, dirs])        # [1, norm, dir]
    lam = 1e-6
    beta = np.linalg.solve(X.T@X + lam*np.eye(3), X.T@alphas)       # β0, β1, β2
    b0, b1, b2 = map(float, beta)

    # Sensitivity scores in [0,1]
    s1, s2 = abs(b1), abs(b2)
    denom  = (s1 + s2) if (s1 + s2) > 0 else 1.0
    sens_norm = float(s1 / denom)
    sens_dir  = float(s2 / denom)

    # Per-region median α
    by_region = defaultdict(list)
    for r in rows:
        by_region[r["region_id"]].append(r["alpha_update"])
    med_by_region = {str(k): float(np.median(v)) for k, v in by_region.items() if v}

    summary = {
        "agg_rule_inference": {
            "sensitivity": {"norm": sens_norm, "direction": sens_dir},
            "beta_raw": {"intercept": b0, "beta_norm": b1, "beta_dir": b2},
            "notes": "Scores in [0,1]; higher means stronger influence on α."
        },
        "per_region": {
            rid: {"median_alpha": med} for rid, med in med_by_region.items()
        },
        "calibration": {"method": "update_lsq"}
    }
    with open(out_json_path, "w") as f:
        json.dump(summary, f, indent=2)
