import csv
import os
from typing import Dict, Iterable, List, Optional

class BackdoorSurvivalTracker:
    def __init__(self, save_dir: str, region_ids: List[int], filename: str = "backdoor_tracking_log.csv"):
        """
        Lightweight per-round tracker: which regions were selected, per-region ASR,
        global clean acc, and malicious participation stats.

        Columns:
          iteration, if_attack (T/F),
          R{rid}_selected, R{rid}_ASR for all rid in region_ids,
          acc, malicious_weight_percent, malicious_client_ratio
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.region_ids = region_ids
        self.filename = filename
        self.csv_path = os.path.join(self.save_dir, self.filename)

        # Put "if_attack" right after iteration (as requested)
        self.fieldnames = (
            ["iteration", "if_attack"]
            + [f"R{rid}_selected" for rid in self.region_ids]
            + [f"R{rid}_ASR" for rid in self.region_ids]
            + ["acc", "malicious_weight_percent", "malicious_client_ratio"]
        )

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log_iter_csv(
        self,
        *,
        iteration: int,
        region_id_to_asr: Dict[int, Optional[float]],
        attacked_regions: Iterable[int],            # regions selected this round (regardless of gating)
        acc: Optional[float] = None,
        malicious_weight_percent: Optional[float] = None,
        malicious_client_ratio: Optional[float] = None,
        if_attack_round: bool = False,              # True if any malicious client actually attacked
    ) -> None:
        attacked_regions = set(attacked_regions)
        entry: Dict[str, object] = {"iteration": iteration, "if_attack": "T" if if_attack_round else "F"}

        for region_id in self.region_ids:
            entry[f"R{region_id}_selected"] = 1 if region_id in attacked_regions else 0
            asr_value = region_id_to_asr.get(region_id)
            entry[f"R{region_id}_ASR"] = round(asr_value, 4) if (asr_value is not None) else 0.0

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
    tracker: BackdoorSurvivalTracker,
    iteration: int,
    global_eval_results: Dict,
    client_region_mapping: Dict[int, int],
    possible_region_ids_list: List[int],
    malicious_weight_percent: Optional[float] = None,
    malicious_client_ratio: Optional[float] = None,
    # Optional extras to compute if_attack
    attack_enable_by_client: Optional[Dict[int, bool]] = None,
    malicious_clients_list: Optional[List[int]] = None,
) -> None:
    """
    Write the per-round summary row.

    if_attack is True iff at least one malicious *virtual* client had attack enabled this round.
    If the gating info isn't provided, it defaults to False (keeps backward compatibility).
    """
    attacked_regions = list(set(client_region_mapping.values()))

    region_id_to_asr = {
        rid: (global_eval_results.get("asr", {}).get(rid, {}).get("asr", None))
        for rid in possible_region_ids_list
    }
    clean_acc = global_eval_results.get("clean_acc", None)

    # Compute if_attack (optional)
    if_attack_round = False
    if attack_enable_by_client is not None and malicious_clients_list is not None:
        if_attack_round = any(bool(attack_enable_by_client.get(mid, False)) for mid in malicious_clients_list)

    tracker.log_iter_csv(
        iteration=iteration,
        region_id_to_asr=region_id_to_asr,
        attacked_regions=attacked_regions,
        acc=clean_acc,
        malicious_weight_percent=malicious_weight_percent,
        malicious_client_ratio=malicious_client_ratio,
        if_attack_round=if_attack_round,
    )
