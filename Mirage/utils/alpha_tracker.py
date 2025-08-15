# utils/alpha_tracker.py
import os, csv

class AlphaTracker:
    def __init__(self, save_dir, filename="alpha_estimates.csv"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, filename)

        # NEW superset header (order chosen for readability)
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

            # discrepancy between sim-view alpha and real-view alpha
            "alpha_gap_abs","alpha_gap_rel",
        ]

        # If an older file exists with a different header, write to *_v2.csv
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, [])
                # compare as sets to allow order drift; if mismatch, switch file
                if set(existing_header) != set(self.fieldnames):
                    base, ext = os.path.splitext(self.path)
                    self.path = f"{base}_v2{ext}"
            except Exception:
                # any read error -> don’t clobber the old file; switch to v2
                base, ext = os.path.splitext(self.path)
                self.path = f"{base}_v2{ext}"

        # Create if missing
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log_many(self, rows):
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            for r in rows:
                # Ensure all columns exist; fill missing with ""
                out = {k: (r.get(k, "")) for k in self.fieldnames}
                w.writerow(out)
