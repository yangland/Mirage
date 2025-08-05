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
            [f"R{rid}_ASR" for rid in self.region_ids] +
            ["acc"]
        )

        # Create file and write header if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()


    def log_iter_csv(self, iteration, region_id_to_asr, attacked_regions, acc=None):
        """
        Immediately log iteration info (append one row to CSV).
        Args:
            iteration (int): Current training round
            region_id_to_asr (dict): Mapping from region ID → ASR (float)
            attacked_regions (list[int]): Region IDs that were attacked this round
            acc (float or None): Accuracy value to log
        """
        entry = {"iteration": iteration}
        for region_id in self.region_ids:
            entry[f"R{region_id}_selected"] = 1 if region_id in attacked_regions else 0

            # Safely get ASR value and handle None
            asr_value = region_id_to_asr.get(region_id)
            entry[f"R{region_id}_ASR"] = round(asr_value, 4) if asr_value is not None else 0.0

        entry["acc"] = round(acc, 4) if acc is not None else ""

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(entry)


def log_backdoor_tracking_csv(
    tracker,
    iteration,
    global_eval_results,
    client_region_mapping,
    possible_region_ids_list
):
    """
    Logs ASR per region and clean accuracy into the CSV via the tracker.

    Args:
        tracker (BackdoorSurvivalTracker): CSV logger instance.
        iteration (int): Current training round.
        global_eval_results (dict): Output of test_global_model().
        client_region_mapping (dict): Mapping from client ID to region ID.
        possible_region_ids_list (list[int]): List of region IDs in consideration.
    """
    # Extract attacked regions from client-region mapping
    attacked_regions = list(set(client_region_mapping.values()))

    # Map region IDs to their ASR values
    region_id_to_asr = {
        rid: global_eval_results["asr"].get(rid, {}).get("asr", None)
        for rid in possible_region_ids_list
    }

    # Clean accuracy
    clean_acc = global_eval_results.get("clean_acc", None)

    # Write to CSV
    tracker.log_iter_csv(
        iteration=iteration,
        region_id_to_asr=region_id_to_asr,
        attacked_regions=attacked_regions,
        acc=clean_acc
    )
