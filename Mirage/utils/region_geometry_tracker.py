# utils/region_geometry_tracker.py
import os, csv

class RegionGeometryTracker:
    def __init__(self, save_dir, filename="region_geometry_per_round.csv"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, filename)
        self.fieldnames = [
            "iteration","region_id",
            "l2_radius","cosine_threshold","update_cone_mode",
            "l2_scale","cos_scale"
        ]
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log_round(self, iteration: int, region_constraints_dict: dict, attacked_regions: list):
        rows = []
        for rid in attacked_regions:
            c = region_constraints_dict.get(rid, {})
            if not c: continue
            rows.append({
                "iteration": iteration,
                "region_id": int(rid),
                "l2_radius": float(c["l2_radius"]),
                "cosine_threshold": float(c["cosine_threshold"]),
                "update_cone_mode": int(c["update_cone_mode"]),
                "l2_scale": float(c.get("l2_scale", float("nan"))),
                "cos_scale": (None if c.get("cos_scale", None) is None else float(c["cos_scale"]))
            })
        if rows:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writerows(rows)
