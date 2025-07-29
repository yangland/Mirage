# utils/backdoor_survival_tracker.py

import csv
import os

class BackdoorSurvivalTracker:
    def __init__(self, save_dir, region_ids):
        """
        Args:
            save_dir (str): Directory to save CSV files.
            region_ids (list[int]): List of region IDs (e.g. [1, 2, 3, 4])
        """
        self.save_dir = save_dir
        self.region_ids = region_ids
        os.makedirs(self.save_dir, exist_ok=True)
        self.records = []

    def log_iteration(self, iteration, region_id_to_asr, attacked_regions):
        """
        Log ASR and whether each region was attacked during this iteration.

        Args:
            iteration (int): Current training round
            region_id_to_asr (dict): Mapping from region ID → ASR (float)
            attacked_regions (list[int]): Region IDs that were attacked this round
        """
        entry = {"iteration": iteration}

        for region_id in self.region_ids:
            entry[f"R{region_id}_selected"] = 1 if region_id in attacked_regions else 0
            entry[f"R{region_id}_ASR"] = region_id_to_asr.get(region_id, 0.0)

        self.records.append(entry)

    def save_csv(self, filename="backdoor_tracking_log.csv"):
        """
        Save all records to a CSV file with column order:
        iteration, R1_selected, R2_selected, ..., Rn_selected, R1_ASR, ..., Rn_ASR
        """
        csv_path = os.path.join(self.save_dir, filename)

        # Separate selected and ASR columns
        selected_fields = [f"R{rid}_selected" for rid in self.region_ids]
        asr_fields = [f"R{rid}_ASR" for rid in self.region_ids]

        fieldnames = ["iteration"] + selected_fields + asr_fields

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.records:
                writer.writerow(entry)

