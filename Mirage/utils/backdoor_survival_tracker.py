# utils/backdoor_survival_tracker.py

import csv
import os
import json

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
            ["acc", "malicious_weight_percent", "malicious_client_ratio"] +
            ["per_client_l2_values", "per_client_cos_values",
             "per_client_l2_scales", "per_client_cos_scales"]
        )


        # Create file and write header if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()


    def log_iter_csv(self, iteration, region_id_to_asr, attacked_regions, acc=None,
                     malicious_weight_percent=None, malicious_client_ratio=None,
                     per_client_l2_values=None, per_client_cos_values=None,          # NEW
                     per_client_l2_scales=None, per_client_cos_scales=None):         # NEW
        entry = {"iteration": iteration}
        for region_id in self.region_ids:
            entry[f"R{region_id}_selected"] = 1 if region_id in attacked_regions else 0
            asr_value = region_id_to_asr.get(region_id)
            entry[f"R{region_id}_ASR"] = round(asr_value, 4) if asr_value is not None else 0.0

        entry["acc"] = round(acc, 4) if acc is not None else ""
        entry["malicious_weight_percent"] = round(malicious_weight_percent, 4) if malicious_weight_percent is not None else ""
        entry["malicious_client_ratio"] = round(malicious_client_ratio, 4) if malicious_client_ratio is not None else ""

        # NEW: store lists-of-lists as JSON strings
        entry["per_client_l2_values"]  = json.dumps(per_client_l2_values  or [])
        entry["per_client_cos_values"] = json.dumps(per_client_cos_values or [])
        entry["per_client_l2_scales"]  = json.dumps(per_client_l2_scales  or [])
        entry["per_client_cos_scales"] = json.dumps(per_client_cos_scales or [])

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(entry)



def log_backdoor_tracking_csv(
    tracker,
    iteration,
    global_eval_results,
    client_region_mapping,
    possible_region_ids_list,
    malicious_weight_percent=None,
    malicious_client_ratio=None,
    region_constraints_dict=None,      # NEW
    selected_clients_list=None         # NEW
):
    # Extract attacked regions from client-region mapping
    attacked_regions = list(set(client_region_mapping.values()))

    # Map region IDs to their ASR values
    region_id_to_asr = {
        rid: global_eval_results["asr"].get(rid, {}).get("asr", None)
        for rid in possible_region_ids_list
    }
    # Clean accuracy
    clean_acc = global_eval_results.get("clean_acc", None)

    # --- NEW: per-client values/scales as lists-of-lists
    per_client_l2_values, per_client_cos_values = [], []
    per_client_l2_scales, per_client_cos_scales = [], []
    if region_constraints_dict and selected_clients_list:
        for cid in selected_clients_list:
            rid = client_region_mapping.get(cid)
            c = region_constraints_dict.get(rid, {})
            if not c:
                continue
            # shape: [client_id, region_id, value]
            per_client_l2_values.append([int(cid), int(rid), float(c["l2_radius"])])
            per_client_cos_values.append([int(cid), int(rid), float(c["cosine_threshold"])])
            per_client_l2_scales.append([int(cid), int(rid), float(c.get("l2_scale", float("nan")))])
            cos_scale_val = c.get("cos_scale", None)
            per_client_cos_scales.append([int(cid), int(rid), None if cos_scale_val is None else float(cos_scale_val)])

    # Write to CSV
    tracker.log_iter_csv(
        iteration=iteration,
        region_id_to_asr=region_id_to_asr,
        attacked_regions=attacked_regions,
        acc=clean_acc,
        malicious_weight_percent=malicious_weight_percent,
        malicious_client_ratio=malicious_client_ratio,
        per_client_l2_values=per_client_l2_values,        # NEW
        per_client_cos_values=per_client_cos_values,      # NEW
        per_client_l2_scales=per_client_l2_scales,        # NEW
        per_client_cos_scales=per_client_cos_scales       # NEW
    )
