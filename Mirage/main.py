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

from utils.utils import args_update, assign_regions_to_malicious
from utils.visualize import visualize_batch
from utils.backdoor_survival_tracker import BackdoorSurvivalTracker

logger = logging.getLogger("logger")


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
    parser.add_argument("--no_of_adversaries", default=3, type=int)
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
        server = No_defense_Server(params=params_loaded, dataloader=dataloader)

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

    all_malicious_clients_ids = [malicious_client for _ in range(params_loaded["no_of_adversaries"])]
    possible_region_ids_list = list(range(1, 5))  # 4 regions
    
    tracker = BackdoorSurvivalTracker(
        save_dir=params_loaded["folder_path"],
        region_ids=possible_region_ids_list
    )


    for iteration in range(server.params["start_iteration"], server.params["end_iteration"]):
        logger.info(f"====================== Current Round: {iteration} ======================")

        # === Step 1: Preprocess + client uploads
        server.pre_process(test_data=server.test_dataloader, iteration=iteration)
        
        # === Step 2: Select clients
        selected_clients, selected_malicious_clients = server.select_clients(iteration)

        # === Step 3: Assign region IDs to malicious clients
        client_region_mapping = assign_regions_to_malicious(
            selected_clients_list=selected_clients,
            malicious_clients_list=selected_malicious_clients,
            iteration=iteration,
            possible_region_ids=possible_region_ids_list,
            server=server
        )
        logger.info(f"[Round {iteration}] Region assignments: {client_region_mapping}")
        
        # === Step 4: Preprocess + client uploads
        server.pre_process(test_data=server.test_dataloader, iteration=iteration)

        (
            weight_accumulator,
            weight_accumulator_by_client,
            aggregated_model_id,
        ) = server.broadcast_upload(
            iteration=iteration,
            benign_client=benign_client,
            malicious_client=malicious_client,
            selected_clients_list=selected_clients,
            malicious_clients_list=selected_malicious_clients,
            client_region_mapping=client_region_mapping 
        )

        # === Step 5: Aggregate model
        print(f"[DEBUG] Global model keys: {list(server.global_model.state_dict().keys())}")
        server.aggregation(agg_method=params_loaded["agg_method"],
                           weight_accumulator_by_client=weight_accumulator_by_client)
        
        logger.info(f"aggregated_model: {aggregated_model_id}")

        # === Step 6: Evaluate global model
        global_eval_results = server.test_global_model(
                                                        iteration=iteration,
                                                        malicious_clients=malicious_client,
                                                        possible_region_ids=possible_region_ids_list,
                                                        client_region_mapping=client_region_mapping
                                                    )
        # log the results in CSV file
        attacked_regions = list(set(client_region_mapping.values()))

        region_id_to_asr = {
            rid: global_eval_results["asr"].get(rid, {}).get("asr", 0.0)
            for rid in possible_region_ids_list
        }

        tracker.log_iteration(
            iteration=iteration,
            region_id_to_asr=region_id_to_asr,
            attacked_regions=attacked_regions
        )
        
        # === Step 7: Save checkpoint
        server.save_model(iteration, malicious_client.trigger_set, malicious_client.mask_set)


# ==================== End of Training ====================
# Save backdoor survival results
tracker.save_csv(filename=f"{params_loaded['folder_path']}/backdoor_tracking_log_{params_loaded['defense_method']}.csv")
