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
    """
    Read per-round α estimates and summarize an aggregator’s preferences.

    Inputs
    -------
    alpha_csv_path : str
        Path to the `alpha_estimates.csv` written by AlphaTracker. Each row
        should minimally include:
          - iteration (int)
          - region_id (int)
          - k_mal (int): number of malicious clients selected in that round
          - P_total (int): total selected clients in that round
          - alpha_update (float): total malicious inclusion share estimated
            from update-space least squares (i.e., g ≈ a*m + b*b)
          - l2_radius (float): region L2 budget used when crafting
          - update_cone_mode (int): 1 = align, otherwise = oppose
    out_json_path : str
        Where to write the summary JSON.

    Method (high level)
    -------------------
    1) We fit a tiny ridge regression on rows:
           alpha_update ≈ β0 + β1 * norm01 + β2 * dirBin
       where:
         - norm01 = min–max normalized `l2_radius` across all rows in the file
         - dirBin = 1 if `update_cone_mode == 1` (align), else 0 (oppose)
       This is not a causal model—just a compact way to quantify how much
       α co-varies with magnitude (L2) vs. direction constraints.

    2) From (β1, β2) we derive normalized sensitivity scores in [0,1] whose
       sum is 1:
           sensitivity.norm  = |β1| / (|β1| + |β2|)
           sensitivity.direction = |β2| / (|β1| + |β2|)
       Higher value ⇒ stronger empirical influence on α in this dataset.

    3) For each region, we aggregate rows (i.e., rounds where that region
       was attacked) into:
         - median_alpha                : median of `alpha_update` (TOTAL malicious share)
         - normalized_median_alpha     : median of (`alpha_update` / k_mal)
           This “per-malicious” α lets you compare experiments with different
           numbers of attackers per round. A useful baseline is the uniform
           FedAvg per-client share 1/P_total. Interpreting:
             • normalized_median_alpha  >>  1/P_total  ⇒ rule tends to overweight
               each malicious client (favorable to attacker)
             • normalized_median_alpha  <<  1/P_total  ⇒ rule tends to downweight
               each malicious client (unfavorable)

    Output JSON schema
    ------------------
    {
      "agg_rule_inference": {
        "sensitivity": { "norm": float in [0,1], "direction": float in [0,1] },
        "beta_raw":    { "intercept": β0, "beta_norm": β1, "beta_dir": β2 },
        "notes": "Scores in [0,1]; higher means stronger influence on α."
      },
      "per_region": {
        "<region_id>": {
          "median_alpha": float,
          "normalized_median_alpha": float
        },
        ...
      },
      "calibration": { "method": "update_lsq" }
    }

    Notes & caveats
    ---------------
    • α (“alpha_update”) is the effective TOTAL share of the malicious centroid
      in the global update for that round; dividing by k_mal yields the
      per-malicious share.
    • The regression is descriptive; coefficients can shift with data scale
      and coverage of (norm, dir) settings.
    • If the CSV has no rows, we write {"error": "no alpha rows"}.
    """
    
    rows = _load_alpha(alpha_csv_path)
    if not rows:
        with open(out_json_path, "w") as f:
            json.dump({"error":"no alpha rows"}, f, indent=2)
        return

    # --- regression features (same as before) ---
    alphas = np.array([r["alpha_update"] for r in rows], dtype=float)
    norms  = _normalize01([r["l2_radius"] for r in rows])
    dirs   = np.array([1.0 if r["update_cone_mode"]==1 else 0.0 for r in rows], dtype=float)

    X = np.column_stack([np.ones_like(alphas), norms, dirs])
    lam = 1e-6
    beta = np.linalg.solve(X.T@X + lam*np.eye(3), X.T@alphas)
    b0, b1, b2 = map(float, beta)

    s1, s2 = abs(b1), abs(b2)
    denom  = (s1 + s2) if (s1 + s2) > 0 else 1.0
    sens_norm = float(s1 / denom)
    sens_dir  = float(s2 / denom)

    # --- per-region medians (total α and per-malicious α) ---
    by_region_total = defaultdict(list)
    by_region_normd = defaultdict(list)
    for r in rows:
        rid = r["region_id"]
        a   = float(r["alpha_update"])
        k   = int(r.get("k_mal", 1)) or 1  # safety
        by_region_total[rid].append(a)
        by_region_normd[rid].append(a / k)

    per_region = {}
    for rid in sorted(by_region_total.keys()):
        med_total = float(np.median(by_region_total[rid])) if by_region_total[rid] else float("nan")
        med_normd = float(np.median(by_region_normd[rid])) if by_region_normd[rid] else float("nan")
        per_region[str(rid)] = {
            "median_alpha": med_total,
            "normalized_median_alpha": med_normd
        }

    summary = {
        "agg_rule_inference": {
            "sensitivity": {"norm": sens_norm, "direction": sens_dir},
            "beta_raw": {"intercept": b0, "beta_norm": b1, "beta_dir": b2},
            "notes": "Scores in [0,1]; higher means stronger influence on α."
        },
        "per_region": per_region,
        "calibration": {"method": "update_lsq"}
    }
    with open(out_json_path, "w") as f:
        json.dump(summary, f, indent=2)
