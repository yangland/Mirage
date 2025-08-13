# utils/alpha_tracker.py
import os, csv

class AlphaTracker:
    def __init__(self, save_dir, filename="alpha_estimates.csv"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, filename)
        self.fieldnames = [
            "iteration","region_id","k_mal","P_total",
            "alpha_update","est_mode","a_raw","b_raw",
            "l2_radius","cosine_threshold","update_cone_mode"
        ]
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log_many(self, rows):
        if not rows: return
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            for r in rows:
                w.writerow(r)
