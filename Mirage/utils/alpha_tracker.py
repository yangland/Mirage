import csv
import os
import math

def _clean_val(v):
    # write "" for NaN (keeps spreadsheets happy), else pass through
    try:
        if isinstance(v, float) and math.isnan(v):
            return ""
    except Exception:
        pass
    return v

class AlphaTracker:
    def __init__(self, save_dir, filename="alpha_estimates.csv"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, filename)

        # Superset header (matches make_round_alpha_rows)
        self.fieldnames = [
            "iteration","region_id","k_mal","P_total",
            # benign diagnostics (sim vs real)
            "b_real_l2","b_sim_l2","b_cos_dist","b_l2_pct_diff",
            # geometry snapshot
            "l2_radius","cosine_threshold","update_cone_mode",
            # alpha (simulated benign reference => malicious-view)
            "alpha_update","est_mode","a_raw","b_raw",
            # alpha (real benign reference => server-view)
            "alpha_update_real","est_mode_real","a_raw_real","b_raw_real",
            # discrepancy (sim-view alpha vs server-view alpha)
            "alpha_gap_abs","alpha_gap_rel",
        ]

        # If an older file exists with a different header (including order), write to *_v2.csv
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, [])
                # compare ORDER-SENSITIVE to avoid misaligned rows
                if existing_header != self.fieldnames:
                    base, ext = os.path.splitext(self.path)
                    self.path = f"{base}_v2{ext}"
            except Exception:
                base, ext = os.path.splitext(self.path)
                self.path = f"{base}_v2{ext}"

        # Create file if missing
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log_many(self, rows):
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            for r in rows:
                # ensure all columns exist; fill missing with ""
                out = {k: _clean_val(r.get(k, "")) for k in self.fieldnames}
                w.writerow(out)
