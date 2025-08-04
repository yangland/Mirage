# utils/backdoor_survival_tracker.py

import csv
import os

class BackdoorSurvivalTracker:
    def __init__(self, save_dir, region_ids, filename="backdoor_tracking_log.csv"):
        """
        Args:
            save_dir (str): Directory to save CSV files.
            region_ids (list[int]): List of region IDs (e.g. [1, 2, 3, 4])
            filename (str): CSV file name
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.region_ids = region_ids
        self.filename = filename
        self.csv_path = os.path.join(self.save_dir, self.filename)

        self.fieldnames = (
            ["iteration"] +
            [f"R{rid}_selected" for rid in self.region_ids] +
            [f"R{rid}_ASR" for rid in self.region_ids]
        )

        # Create file and write header if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_iter_csv(self, iteration, region_id_to_asr, attacked_regions):
        """
        Immediately log iteration info (append one row to CSV).
        Args:
            iteration (int): Current training round
            region_id_to_asr (dict): Mapping from region ID → ASR (float)
            attacked_regions (list[int]): Region IDs that were attacked this round
        """
        entry = {"iteration": iteration}
        for region_id in self.region_ids:
            entry[f"R{region_id}_selected"] = 1 if region_id in attacked_regions else 0
            entry[f"R{region_id}_ASR"] = round(region_id_to_asr.get(region_id, 0.0), 4)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(entry)


