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

from utils.utils import args_update, model_dist_norm_var, poisoned_batch_injection, evaluate_asr_before_aggregation, evaluate_asr_after_aggregation
from utils.visualize import visualize_batch
from utils.backdoor_survival_tracker import BackdoorSurvivalTracker
tracker = BackdoorSurvivalTracker(save_dir="results/backdoor_tracking")

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
        malicious_client = MirageClient(params_loaded, dataloader.train_dataloader, dataloader.test_dataloader)

    else:
        malicious_client = MaliciousClient(params_loaded, dataloader.train_dataloader, dataloader.test_dataloader)


    prev_asr_before = None  # Stores ASR before aggregation from previous round

    for iteration in range(server.params["start_iteration"], server.params["end_iteration"]):
        logger.info(f"====================== Current Round: {iteration} ======================")

        # === Step 1: Evaluate ASR of server model BEFORE crafting new attacks (i.e., survival from last round)
        if prev_asr_before is not None:
            avg_asr_after = evaluate_asr_after_aggregation(
                server=server,
                region_assignments=region_assignments,
                malicious_client=malicious_client,
                params=params_loaded
            )
            tracker.log_iteration(iteration, prev_asr_before, avg_asr_after)

        # === Step 2: Preprocess + client uploads
        server.pre_process(test_data=server.test_dataloader, iteration=iteration)

        (
            weight_accumulator,
            weight_accumulator_by_client,
            aggregated_model_id,
            region_assignments,
            malicious_models_by_id
        ) = server.broadcast_upload(
            iteration=iteration,
            benign_client=benign_client,
            malicious_client=malicious_client
        )

        # === Step 3: Aggregate model
        server.aggregation(weight_accumulator=weight_accumulator, aggregated_model_id=aggregated_model_id)
        logger.info(f"aggregated_model: {aggregated_model_id}")

        # === Step 4: Evaluate global model
        server.test_global_model(iteration=iteration, malicious_clients=malicious_client)

        # === Step 5: Save checkpoint
        server.save_model(iteration, malicious_client.trigger_set, malicious_client.mask_set)

        # === Step 6: Evaluate ASR before aggregation (i.e., attack strength of current round)
        prev_asr_before = evaluate_asr_before_aggregation(
            server=server,
            malicious_models_by_id=malicious_models_by_id,
            region_assignments=region_assignments,
            malicious_client=malicious_client,
            params=params_loaded
        )


# ==================== End of Training ====================
# Save backdoor survival results
tracker.save_csv(filename=f"backdoor_survival_log_{params_loaded['defense_method']}.csv")
tracker.summarize_preferences()
