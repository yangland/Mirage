import copy
import random

import numpy as np
import yaml
import pickle
import torch
import logging
import argparse
from colorama import Fore
from torchsummary import summary

from datasets.MSP_dataloader import MSPDataloader

from participants.clients.BenignClient import BenignClient
from participants.clients.MalicilousClient import MaliciousClient
from participants.clients.MirageClient import MirageClient
from participants.servers.No_defense_Server import No_defense_Server

from utils.utils import args_update, assign_regions_to_malicious, poisoned_batch_injection,\
    virtual_mali_id_assignment, get_regions_to_attack, analyze_malicious_contribution, test_model_asr_acc,\
    _tiny_fp, build_pooled_malicious_test_loader
from utils.visualize import visualize_batch
from utils.backdoor_survival_tracker import BackdoorSurvivalTracker, log_backdoor_tracking_csv
from utils.alpha_tracker import AlphaTracker
from utils.region_geometry_tracker import RegionGeometryTracker
from utils.alpha_estimation import make_round_alpha_rows
from utils.agg_pref_summary import summarize_preferences
import os

logger = logging.getLogger("logger")

# Fixed region-to-client mapping — e.g., {region_id → canonical_client_id}
# Defines which client holds the canonical trigger/mask for each region. Used to initialize trigger_set_by_region
canonical_client_for_region = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

"""
canonical_client_for_region:            region_id → canonical_client_id
↓ Used to initialize triggers
malicious_client.trigger_set_by_region: region_id → trigger

client_region_mapping:                  client_id → region_id
↓ Used to update region_to_test_client
region_to_test_client:                  region_id → client_id
↓ Inverted
reverse_client_region_mapping:          client_id → region_id
"""

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    yaml_file = "yamls/Mirage/Mirage_nodefense.yaml"
    parser.add_argument("--params", default=f"{yaml_file}", type=str)
    # parser.add_argument("--no_of_total_adversaries", default=3, type=int)
    #  full_random  sequential_poison continue_poison
    parser.add_argument("--poison_type", default="continue_poison", type=str)
    parser.add_argument("--attach", default="", type=str)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--model_type", default="ResNet18", type=str)
    parser.add_argument("--dataset", default="CIFAR10", type=str)

    args = parser.parse_args()
    params_loaded = args_update(args)

    if params_loaded["dataset"].upper() == "CIFAR10":
        params_loaded["class_num"] = 10
    elif params_loaded["dataset"].upper() == "CIFAR100":
        params_loaded["class_num"] = 100
    elif params_loaded["dataset"].upper() == "EMNIST":
        params_loaded["class_num"] = 10
    elif params_loaded["dataset"].upper() == "GTSRB":
        params_loaded["class_num"] = 43
        params_loaded["poison_train_batch_size"] = 32
        params_loaded["train_batch_size"] = 32
        params_loaded["poisoned_len"] = 4
    else:
        raise NotImplementedError


    logger.info(f'params_loaded["resumed_model"] - {params_loaded["resumed_model"]}')

    logger.info(f"Params: {params_loaded}")
    set_random_seed(params_loaded["seed"])

    dataloader = MSPDataloader(params_loaded)

    server = None
    if params_loaded["defense_method"].lower() == "nodefense":
        server = No_defense_Server(
                params=params_loaded,
                dataloader=dataloader,
                full_train_dataset=dataloader.train_dataset
            )
        server.total_malicious_clients = list(range(params_loaded["no_of_total_adversaries"]))
    else:
        raise NotImplementedError

    benign_client = BenignClient(params_loaded, dataloader.train_dataloader, dataloader.test_dataloader)

    if params_loaded["malicious_train_algo"] == "Mirage":
        malicious_client = MirageClient(params_loaded, 
                                        dataloader.train_dataloader, 
                                        dataloader.test_dataloader)
    else:
        malicious_client = MaliciousClient(params_loaded, 
                                           dataloader.train_dataloader, 
                                           dataloader.test_dataloader)
        
    print(f"[DEBUG] Type of malicious_client: {type(malicious_client)}")
    mali_test_loader = build_pooled_malicious_test_loader(
        malicious_client, server.total_malicious_clients, params_loaded["test_batch_size"]
    )

    possible_region_ids_list = list(range(1, 5))  # 4 regions
    # region_to_test_client {region_id → client_id}
    # Maps region to the specific client used to test ASR on that region
    region_to_test_client = {}
    for rid in malicious_client.trigger_set_by_region.keys():
        if rid not in region_to_test_client:
            region_to_test_client[rid] = canonical_client_for_region[rid]
    
    logger.info(f"Initial region to test client mapping: {region_to_test_client}")

    main_tracker = BackdoorSurvivalTracker(
        save_dir=params_loaded["folder_path"],
        region_ids=possible_region_ids_list,
        filename=f"backdoor_tracking_log_{params_loaded['defense_method']}.csv"
    )
    
    alpha_tracker = AlphaTracker(save_dir=params_loaded["folder_path"])
    geom_tracker  = RegionGeometryTracker(save_dir=params_loaded["folder_path"])
    
    # === State for cross-iteration attack control ===
    last_global_eval = {
        "clean_acc": None,
        "asr_by_region": {rid: 0.0 for rid in possible_region_ids_list}
    }
    
    iteration=-100
    
    for iteration in range(server.params["start_iteration"], server.params["end_iteration"]):
        logger.info(f"====================== Current Round: {iteration} ======================")

        # === Step 1: Preprocess + client uploads
        server.pre_process(test_data=server.test_dataloader, iteration=iteration)
        logger.info(f"[Round {iteration}] Aggregation rule: {server.params['agg_method']}, ")
        
        # === Step 2: Select clients
        selected_clients, selected_malicious_clients = server.select_clients(
                                                                iteration=iteration,
                                                                region_to_malicious_client=canonical_client_for_region
                                                            )
        # Original malicious clients (e.g., [2, 2, 2, 2])
        # map to [22001,22002,22003,22004]
        # Step 2.5: Assign virtual malicious IDs and regions
        regions_to_attack = get_regions_to_attack(
                selected_malicious_clients=selected_malicious_clients,
                canonical_client_for_region=canonical_client_for_region
            )

        (
            virtual_malicious_clients,
            malicious_client_mapping,
            client_region_mapping
        ) = virtual_mali_id_assignment(
            selected_malicious_clients=selected_malicious_clients,
            regions_to_attack=regions_to_attack,
            virtual_id_base=20000
        )
        logger.info(f"[Round {iteration}] Virtual malicious clients: {virtual_malicious_clients}")
        # logger.info(f"[Round {iteration}] Malicious client mapping: {malicious_client_mapping}")
        # logger.info(f"[Round {iteration}] Client region mapping: {client_region_mapping}")

        # Add benign clients
        selected_benign_clients = [c for c in selected_clients if c not in selected_malicious_clients]
        # Final selected clients list
        selected_clients = virtual_malicious_clients + selected_benign_clients
        
        # === Step 3: Assign region IDs to malicious clients
        # client_region_mapping {client_id → region_id}
        # Maps selected malicious clients to the regions they're attacking in the current iteration
        client_region_mapping = assign_regions_to_malicious(
            selected_clients_list=selected_clients,
            malicious_clients_list=virtual_malicious_clients,
            iteration=iteration,
            possible_region_ids=possible_region_ids_list,
            server=server,
            strategy=params_loaded.get("clients_region_map", "by_order"),
            predefined_id_set=params_loaded.get("predefined_id_set", None)
        )
        
        # === Attack gating (per malicious virtual client) ===
        round_idx = iteration - server.params["start_iteration"]
        warmup_len = int(params_loaded.get("warmup_no_attack_iters", 0))  # default 0 = no warmup
        in_warmup = round_idx < warmup_len

        attack_min_acc = params_loaded.get("attack_min_clean_acc", 0.0)
        skip_if_region_asr_ge = params_loaded.get("skip_attack_if_region_asr_ge", 2.0)  # 2.0 ~ disabled

        attack_enable_by_client = {}
        for virtual_id in virtual_malicious_clients:
            region_id = client_region_mapping.get(virtual_id)
            attack = not in_warmup

            # Gate by global clean acc (use last *malicious-view* eval if available)
            if last_global_eval["clean_acc"] is not None and last_global_eval["clean_acc"] < attack_min_acc:
                attack = False

            # Gate by region ASR
            prev_asr = last_global_eval["asr_by_region"].get(region_id, 0.0)
            if prev_asr >= skip_if_region_asr_ge:
                attack = False

            attack_enable_by_client[virtual_id] = attack
        
        if_attack_round = any(bool(v) for v in attack_enable_by_client.values())
        logger.info(f"[Round {iteration}] Attack gating per malicious client: {attack_enable_by_client}")


        # === Step 4: Broadcast model to clients (including poisoned clients)
        (
            weight_accumulator,
            weight_accumulator_by_client,
            aggregated_model_id,
            region_constraints_dict,
            sim_benign_avg_update,
            real_benign_avg_update,
        ) = server.broadcast_upload(
            iteration=iteration,
            benign_client=benign_client,
            malicious_client=malicious_client,
            selected_clients_list=selected_clients,                 # e.g. [2000, 2001, 32, 46, ...]
            malicious_clients_list=virtual_malicious_clients,       # e.g. [2000, 2001, ...]
            client_region_mapping=client_region_mapping,            # e.g. {2000: 1, 2001: 1, ...}
            canonical_client_for_region=canonical_client_for_region,  # ✅ FIXED: should map region_id → canonical_client_id
            malicious_client_mapping=malicious_client_mapping,       # ✅ virtual_id → real_id
            attack_enable_by_client=attack_enable_by_client,
            mali_test_loader=mali_test_loader,
        )
        
        # Ensure every region that has a trigger is mapped to a test client (warm-up included)
        for rid in malicious_client.trigger_set_by_region.keys():
            if rid not in region_to_test_client:
                region_to_test_client[rid] = canonical_client_for_region[rid]

        # === Step 5: Aggregate model
        prev_global_sd = {k: v.detach().clone() for k, v in server.global_model.state_dict().items()}
        client_weights = server.aggregation(
            agg_method=params_loaded["agg_method"],
            weight_accumulator_by_client=weight_accumulator_by_client
            )
        logger.info(f"[Round {iteration}] Client weights: {client_weights}")
        
        # 1) Alpha estimates (update-space)
        alpha_rows = make_round_alpha_rows(
            iteration=iteration,
            prev_global_sd=prev_global_sd,
            new_global_sd=server.global_model.state_dict(),
            updates_by_client=weight_accumulator_by_client,
            selected_clients=selected_clients,
            malicious_clients=virtual_malicious_clients,
            client_region_mapping=client_region_mapping,
            region_constraints_dict=region_constraints_dict,
            sim_benign_avg_update=sim_benign_avg_update,
            real_benign_avg_update=real_benign_avg_update,
            attack_enable_by_client=attack_enable_by_client,
            include_disabled=False,
        )
        alpha_tracker.log_many(alpha_rows)

        # 2) Geometry CSV (only for attacked regions this round)
        # ONLY regions corresponding to malicious virtual clients this round
        attacked_regions = sorted({
            rid for cid, rid in client_region_mapping.items()
            if cid in virtual_malicious_clients
        })
        geom_tracker.log_round(iteration, region_constraints_dict, attacked_regions)


        # Analyze malicious weight
        malicious_stats = analyze_malicious_contribution(
            client_weights=client_weights,
            selected_clients=selected_clients,
            selected_malicious_clients=virtual_malicious_clients,
            logger=logger
        )

        # === Update region→client mapping for ASR testing
        for client_id, region_id in client_region_mapping.items():
            if region_id in malicious_client.trigger_set_by_region:
                real_client_id = malicious_client_mapping.get(client_id, client_id)
                region_to_test_client[region_id] = real_client_id


        # === Rebuild reverse mapping to be passed to test_global_model
        reverse_client_region_mapping = {
            client_id: region_id
            for region_id, client_id in region_to_test_client.items()}

        logger.info(f"Updated reverse mapping: {reverse_client_region_mapping}")


        # === Step 6: Evaluate global model
        # Server view (clean test set)
        global_eval = server.test_global_model(
            iteration=iteration,
            malicious_clients=malicious_client,
            possible_region_ids=possible_region_ids_list,
            client_region_mapping=reverse_client_region_mapping,
            server_test_loader=None,             # defaults to server.test_dataloader
            mali_test_loader=mali_test_loader,   # pooled malicious set
            log_views="both",
            log_trigger_dbg=params_loaded.get("log_trigger_dbg", False),
            show_tsne=params_loaded.get("show_tsne", False),
        )

        server_view = global_eval["server"]
        mali_view   = global_eval["malicious"]

        # Use malicious view for gating next round
        last_global_eval = {
            "clean_acc": mali_view["clean_acc"] if mali_view else None,
            "asr_by_region": {
                rid: (mali_view["asr"].get(rid, {}).get("asr", 0.0) if mali_view else 0.0)
                for rid in possible_region_ids_list
            }
        }

        logger.info(f"[Round {iteration}] Last global eval (for gating next round): "
                    f"acc={last_global_eval['clean_acc']}, "
                    f"ASR_by_region={ {k: round(v*100,2) for k,v in last_global_eval['asr_by_region'].items()} }")

                
        # log the results in CSV file
        log_backdoor_tracking_csv(
            tracker=main_tracker,
            iteration=iteration,
            global_eval_results=server_view,      # or mali_view; whichever you want to chart
            client_region_mapping=client_region_mapping,
            possible_region_ids_list=possible_region_ids_list,
            malicious_weight_percent=malicious_stats["malicious_weight_percent"],
            malicious_client_ratio=malicious_stats["malicious_client_ratio"],
            attack_enable_by_client=attack_enable_by_client,          # NEW
            malicious_clients_list=virtual_malicious_clients,         # NEW
        )        

        # === Step 7: Save checkpoint
        server.save_model(iteration, malicious_client.trigger_set, malicious_client.mask_set)

alpha_csv_path = alpha_tracker.path
out_json_path  = os.path.join(params_loaded["folder_path"], "agg_pref_summary.json")
summarize_preferences(alpha_csv_path, out_json_path)
logger.info(f"[Summary] Wrote aggregation preference JSON → {out_json_path}")

logger.info(f"Round {iteration} completed - FL finished.")



