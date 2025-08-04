import copy
import datetime
import itertools
import logging
import os

import math
import shutil  # Add this import at the top if not already there
import numpy as np
import torch
import yaml
from colorama import Fore

logger = logging.getLogger("logger")

# 创建一个全局变量static_args，用于保存命令行参数
global static_args
static_args = None


def args_update(args=None, mkdir=True):
    with open(f"./{args.params}", "r", encoding="utf-8") as f:
        new_args = yaml.safe_load(f)
    # new_args.update(vars(args))
    
    # ✅ only add args from CLI if not already in YAML
    for k, v in vars(args).items():
        if k not in new_args:
            new_args[k] = v

    if "run_device" in new_args:
        run_device = torch.device(new_args["run_device"])
    else:
        if torch.cuda.is_available():
            run_device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            run_device = torch.device("mps")
        else:
            run_device = torch.device("cpu")

    new_args["run_device"] = run_device
    print(Fore.GREEN + f"Running on device: {run_device}" + Fore.RESET)

    new_args["run_device"] = run_device
    # 保留位数的默认处理
    new_args["round_ndigits"] = 8 if "round_ndigits" not in new_args.keys() else new_args[
        "round_ndigits"]

    print(Fore.GREEN + f"Running on device: {run_device}" + Fore.RESET)

    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    agg_method = new_args["agg_method"].lower()
    new_args["folder_path"] = f"./saved_logs/{new_args['attach']}_{current_time}_{agg_method}"
    new_args["save_on_iteration"].append(new_args["end_iteration"] - 1)
    if mkdir:
        try:
            os.makedirs(new_args["folder_path"])
        except FileExistsError:
            logger.info("Folder already exists")
        logger.addHandler(logging.FileHandler(filename=f"{new_args['folder_path']}/log.txt"))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.info(f"current path:{new_args['folder_path']}")
        
        # Copy the YAML file into the folder path for reference
        yaml_dest_path = os.path.join(new_args["folder_path"], os.path.basename(args.params))
        shutil.copy(f"./{args.params}", yaml_dest_path)
        logger.info(f"Copied YAML config to: {yaml_dest_path}")

    # 给全局变量赋值
    global static_args
    static_args = new_args
    return new_args


def poisoned_batch_injection(batch, trigger, mask, is_eval=False, client_id=0, region_id=None, mode = "all"):
    '''
    对batch数据进行投毒，并返回新的batch数据

    :param batch: 需要投毒的batch
    :param trigger: trigger tensor, shape为(channel, height, width)
    :param mask: mask tensor, shape为(channel, height, width)
    :param is_eval: 是否为eval模式
    :param client_id: 客户端id (for attacker)
    :param label_swap: target label (for attacker)
    :param mode: 模式，是否在注入时排除干净目标类, value: "all"/"escape_clean"
    :return: poisoned batch
    '''
    # if label_swap is None:
    #     label_swap = static_args["poison_label_swap"][client_id]
    
    # update the label_swap based on region_mapping
    label_swap = static_args["poison_label_swap_by_region"][region_id]

    
    data, label = copy.deepcopy(batch)

    if mode == "all" and is_eval == False:
        poison_indices = list(range(static_args["poisoned_len"]))
    elif mode == "escape_clean" and is_eval == False:
        poison_indices = np.nonzero(label != label_swap)[:static_args["poisoned_len"]].ravel()
    elif is_eval == True:
        # 生成一个长度为len(label)的列表

        poison_indices = list(range(len(label)))
    else:
        raise ValueError("mode should be 'all' or 'escape_clean'")


    # 如果model == "escap_clean"， 那么poison_indices 为clean_indices中前poisoned_len个不为true的索引

    poisoned_len = static_args["poisoned_len"] if not is_eval else len(label)
    data = data.to(static_args["run_device"])
    trigger = trigger.to(static_args["run_device"])
    mask = mask.to(static_args["run_device"])
    if static_args["poisoned_pattern_choose"] == 1:  # 1 -> pixel block
        data[poison_indices] = trigger * mask + (1 - mask) * data[poison_indices]
    elif static_args["poisoned_pattern_choose"] == 2:  # 1 -> blend trigger
        data[poison_indices] = trigger * static_args["blend_alpha"] + (1 - static_args["blend_alpha"]) * data[
                                                                                                        poison_indices]
    label[poison_indices] = label_swap
    return data, label


def model_dist_norm_var(model, target_params_variables, norm=2):
    '''
    计算model 和 target_params_variables之间的距离，默认使用2范数
    :param self:
    :param model:
    :param target_params_variables:
    :param norm:
    :return: model和target_params_variables之间的距离
    '''
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.zeros(size, device=static_args["run_device"], dtype=torch.float)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)


def model_dist_norm(model, target_params):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
    return math.sqrt(squared_sum)


def update_weight_accumulator_old(model, global_model, weight_accumulator, weight = 1.0):
    '''
    计算模型更新的梯度，并累加到weight_accumulator中


    :param model:
    :param global_model:
    :param weight_accumulator:
    :return: weight_accumulator (当前权重累加), single_weight_accumulator (当前模型更新)
    '''
    single_weight_accumulator = dict()
    for name, data in model.state_dict().items():
        single_weight_accumulator[name] = data - global_model.state_dict()[name]
        try:
            weight_accumulator[name].add_((data - global_model.state_dict()[name]) * weight)
        except RuntimeError as e:
            if single_weight_accumulator[name].dtype == torch.int64:
                weight_accumulator[name].add_((data - global_model.state_dict()[name]))
    return weight_accumulator, single_weight_accumulator


def update_weight_accumulator(model, global_model, weight_accumulator, weight=1.0):
    '''
    Compute model updates and accumulate them into weight_accumulator.

    Keeps all entries in the state_dict, including int and float tensors,
    but skips non-tensor entries to ensure valid accumulation.

    Returns:
        - weight_accumulator (updated global accumulator)
        - single_weight_accumulator (per-client update)
    '''
    single_weight_accumulator = dict()
    model_state = model.state_dict()
    global_state = global_model.state_dict()

    for name, data in model_state.items():
        global_data = global_state[name]

        # Skip non-tensor parameters
        if not isinstance(data, torch.Tensor) or not isinstance(global_data, torch.Tensor):
            print(f"[WARNING] Skipping non-tensor param: {name} (type: {type(data)})")
            continue

        delta = data - global_data

        # Optional: cast to float32 if needed
        if not torch.is_floating_point(delta):
            delta = delta.to(torch.float32)

        single_weight_accumulator[name] = delta.clone()

        if name not in weight_accumulator:
            weight_accumulator[name] = torch.zeros_like(delta)

        try:
            weight_accumulator[name] += delta * weight
        except RuntimeError:
            delta_scaled = (delta * weight).to(weight_accumulator[name].dtype)
            weight_accumulator[name] += delta_scaled

    return weight_accumulator, single_weight_accumulator


# def evaluate_asr_before_aggregation(server, malicious_models_by_id, region_assignments, malicious_client, params):
#     """
#     Compute ASR before aggregation for each region.
#     Returns: dict {region_id: avg_asr}
#     """
#     region_id_to_asrs = {}

#     for client_id, model in malicious_models_by_id.items():
#         region_id = region_assignments[client_id]
#         trigger = malicious_client.trigger_set[client_id]
#         mask = malicious_client.mask_set[client_id]
#         label_swap = params["poison_label_swap"][client_id]

#         asr, _ = server.test_model_once(
#             iteration=None,
#             test_dataloader=server.test_dataloader,
#             is_poisoned=True,
#             model=model,
#             trigger=trigger,
#             mask=mask,
#             label_swap=label_swap
#         )

#         region_id_to_asrs.setdefault(region_id, []).append(asr)

#     # Average ASR per region
#     avg_asr_before = {
#         region_id: sum(asrs) / len(asrs) if len(asrs) > 0 else 0.0
#         for region_id, asrs in region_id_to_asrs.items()
#     }

#     return avg_asr_before


# def evaluate_asr_after_aggregation(server, region_assignments, malicious_client, params):
#     """
#     Compute ASR after aggregation for each region.
#     Returns: dict {region_id: avg_asr}
#     """
#     region_id_to_asrs = {}

#     for client_id, region_id in region_assignments.items():
#         trigger = malicious_client.trigger_set[client_id]
#         mask = malicious_client.mask_set[client_id]
#         label_swap = params["poison_label_swap"][client_id]

#         asr, _ = server.test_model_once(
#             iteration=None,
#             test_dataloader=server.test_dataloader,
#             is_poisoned=True,
#             model=None,  # use global model
#             trigger=trigger,
#             mask=mask,
#             label_swap=label_swap
#         )

#         region_id_to_asrs.setdefault(region_id, []).append(asr)

#     # Average ASR per region
#     avg_asr_after = {
#         region_id: sum(asrs) / len(asrs) if len(asrs) > 0 else 0.0
#         for region_id, asrs in region_id_to_asrs.items()
#     }

#     return avg_asr_after


def assign_regions_to_malicious(selected_clients_list, malicious_clients_list,
                                iteration, possible_region_ids=[1, 2, 3, 4], server=None):
    region_assignments = {}
    ids = possible_region_ids
    num_regions = len(ids)
    malicious_clients_this_round = [cid for cid in selected_clients_list if cid in malicious_clients_list]

    for client_id in malicious_clients_this_round:
        region_id = ids[server.region_index % num_regions]
        region_assignments[client_id] = region_id
        logger.info(f"[Round {iteration}] Assigned Client {client_id} to Region {region_id}")
        server.region_index += 1  # Increment after each assignment

    return region_assignments