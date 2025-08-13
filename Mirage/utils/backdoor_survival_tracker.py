# utils/backdoor_survival_tracker.py

import csv
import os

class BackdoorSurvivalTracker:
    def __init__(self, save_dir, region_ids, filename="backdoor_tracking_log.csv"):
        """
        Lightweight per-round tracker: which regions were selected, per-region ASR,
        global clean acc, and malicious participation stats.
        (Detailed geometry is tracked elsewhere.)
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.region_ids = region_ids
        self.filename = filename
        self.csv_path = os.path.join(self.save_dir, self.filename)

        # NOTE: no per_client_* columns here anymore
        self.fieldnames = (
            ["iteration"] +
            [f"R{rid}_selected" for rid in self.region_ids] +
            [f"R{rid}_ASR" for rid in self.region_ids] +
            ["acc", "malicious_weight_percent", "malicious_client_ratio"]
        )

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log_iter_csv(
        self,
        iteration,
        region_id_to_asr,
        attacked_regions,
        acc=None,
        malicious_weight_percent=None,
        malicious_client_ratio=None,
    ):
        entry = {"iteration": iteration}

        for region_id in self.region_ids:
            entry[f"R{region_id}_selected"] = 1 if region_id in attacked_regions else 0
            asr_value = region_id_to_asr.get(region_id)
            entry[f"R{region_id}_ASR"] = round(asr_value, 4) if asr_value is not None else 0.0

        entry["acc"] = round(acc, 4) if acc is not None else ""
        entry["malicious_weight_percent"] = (
            round(malicious_weight_percent, 4) if malicious_weight_percent is not None else ""
        )
        entry["malicious_client_ratio"] = (
            round(malicious_client_ratio, 4) if malicious_client_ratio is not None else ""
        )

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(entry)


def log_backdoor_tracking_csv(
    tracker,
    iteration,
    global_eval_results,
    client_region_mapping,
    possible_region_ids_list,
    malicious_weight_percent=None,
    malicious_client_ratio=None,
):
    """Write the slim per-round row (no detailed geometry)."""
    attacked_regions = list(set(client_region_mapping.values()))

    region_id_to_asr = {
        rid: global_eval_results["asr"].get(rid, {}).get("asr", None)
        for rid in possible_region_ids_list
    }
    clean_acc = global_eval_results.get("clean_acc", None)

    tracker.log_iter_csv(
        iteration=iteration,
        region_id_to_asr=region_id_to_asr,
        attacked_regions=attacked_regions,
        acc=clean_acc,
        malicious_weight_percent=malicious_weight_percent,
        malicious_client_ratio=malicious_client_ratio,
    )
