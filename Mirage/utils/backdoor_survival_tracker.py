# utils/backdoor_survival_tracker.py

import csv
import os
from collections import defaultdict

class BackdoorSurvivalTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Store per-iteration records: region_id -> list of (before_asr, after_asr)
        self.records = defaultdict(list)

    def log_iteration(self, iteration, region_id_to_asr_before, region_id_to_asr_after):
        """
        Save ASR before and after aggregation for each region at a given round.
        """
        for region_id in region_id_to_asr_before:
            before = region_id_to_asr_before[region_id]
            after = region_id_to_asr_after.get(region_id, 0.0)
            self.records[region_id].append({
                "iteration": iteration,
                "before_asr": before,
                "after_asr": after,
                "survival_rate": after / before if before > 0 else 0.0
            })


    def save_csv(self, filename="backdoor_survival_log.csv"):
        """
        Save the full log of ASRs and survival rates to CSV.
        """
        csv_path = os.path.join(self.save_dir, filename)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["region_id", "iteration", "before_asr", "after_asr", "survival_rate"])
            writer.writeheader()
            for region_id, entries in self.records.items():
                for entry in entries:
                    writer.writerow({
                        "region_id": region_id,
                        **entry
                    })

    def summarize_preferences(self, filename="region_preference_summary.csv"):
        """
        Compute average survival rate per region and save.
        """
        summary_path = os.path.join(self.save_dir, filename)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["region_id", "avg_survival_rate", "num_rounds"])
            writer.writeheader()
            for region_id, entries in self.records.items():
                rates = [entry["survival_rate"] for entry in entries]
                avg_rate = sum(rates) / len(rates) if rates else 0.0
                writer.writerow({
                    "region_id": region_id,
                    "avg_survival_rate": avg_rate,
                    "num_rounds": len(entries)
                })
