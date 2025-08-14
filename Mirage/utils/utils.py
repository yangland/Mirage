import copy
import datetime
import itertools
import logging
import os
import random

import math
import shutil  # Add this import at the top if not already there
import numpy as np
import torch
import yaml
from colorama import Fore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.regoin_utils import flatten_model, check_cos_constraint

from collections.abc import Mapping, Sequence
import torch
from torch.utils.data import DataLoader, ConcatDataset

logger = logging.getLogger("logger")

# 创建一个全局变量static_args，用于保存命令行参数
global static_args
static_args = None


def _tiny_fp(x, n=8):
    try:
        return float(x.reshape(-1)[:n].abs().sum().item())
    except Exception:
        return float('nan')


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
    new_args["folder_path"] = f"./savedloggers/{new_args['attach']}_{current_time}_{agg_method}"
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


def update_weight_accumulator_direct(single_wa, accumulator):
    for key in accumulator:
        accumulator[key] += single_wa[key]
    return accumulator


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


def assign_regions_to_malicious(
    selected_clients_list,
    malicious_clients_list,
    iteration,
    possible_region_ids,
    server,
    strategy="by_order",
    predefined_id_set=None
):
    """
    Assign region IDs to selected malicious clients using different strategies.

    Returns:
        dict[int, int]: Mapping from malicious client ID to region ID.
    """
    region_assignments = {}
    ids = possible_region_ids
    num_regions = len(ids)

    # Filter clients that are actually malicious this round
    malicious_clients_this_round = [cid for cid in selected_clients_list if cid in malicious_clients_list]

    if strategy == "by_order":
        for client_id in malicious_clients_this_round:
            region_id = ids[server.region_index % num_regions]
            region_assignments[client_id] = region_id
            logger.info(f"[Round {iteration}] Assigned Client {client_id} to Region {region_id} (by_order)")
            server.region_index += 1

    elif strategy == "random":
        region_choices = random.sample(ids, len(malicious_clients_this_round))
        for client_id, region_id in zip(malicious_clients_this_round, region_choices):
            region_assignments[client_id] = region_id
            logger.info(f"[Round {iteration}] Assigned Client {client_id} to Region {region_id} (random)")

    elif strategy == "pre_defined":
        if predefined_id_set is None:
            raise ValueError("predefined_id_set must be provided for 'pre_defined' strategy.")

        round_idx = (iteration - server.params["start_iteration"]) % len(predefined_id_set)

        if round_idx >= len(predefined_id_set):
            raise ValueError(f"predefined_id_set only has {len(predefined_id_set)} entries, but got round {round_idx}")

        regions_to_assign = predefined_id_set[round_idx]
        num_regions_to_assign = len(regions_to_assign)
        num_malicious = len(malicious_clients_this_round)

        # Check if regions can be evenly assigned
        if num_malicious % num_regions_to_assign != 0:
            raise ValueError(
                f"Cannot evenly assign {num_malicious} malicious clients to "
                f"{num_regions_to_assign} regions in predefined_id_set[{round_idx}]"
            )

        # Repeat region IDs to match number of malicious clients
        repeat_factor = num_malicious // num_regions_to_assign
        expanded_region_ids = regions_to_assign * repeat_factor

        for client_id, region_id in zip(malicious_clients_this_round, expanded_region_ids):
            region_assignments[client_id] = region_id
            logger.info(f"[Round {iteration}] Assigned Client {client_id} to Region {region_id} (pre_defined)")

    else:
        raise ValueError(f"Unknown clients_region_map strategy: {strategy}")

    return region_assignments

def model_weight_diff(after, before):
    """
    Compute the element-wise difference between two model state_dicts.
    """
    return {k: after[k] - before[k] for k in after}


def grad_weighted_sum(grad_dict, weights_dict):
    """
    Weighted sum of gradients.
    :param grad_dict: dict of client_id -> state_dict
    :param weights_dict: dict of client_id -> scalar weight
    :return: aggregated gradient (state_dict)
    """
    agg_grad = {}
    for client_id, grad in grad_dict.items():
        weight = float(weights_dict.get(client_id, 0.0))  # Ensure scalar float
        for k, v in grad.items():
            weighted = v.clone().detach() * weight
            if k not in agg_grad:
                agg_grad[k] = weighted
            else:
                if weighted.shape != agg_grad[k].shape:
                    weighted = weighted.reshape_as(agg_grad[k])  # ✅ match shape
                agg_grad[k] += weighted
    return agg_grad



def has_non_finite_tensor(state_dict, name="tensor"):
    for k, v in state_dict.items():
        if not torch.isfinite(v).all():
            print(f"[WARNING] Non-finite value detected in {name} at {k}")
            return True
    return False



def virtual_mali_id_assignment(selected_malicious_clients, regions_to_attack, virtual_id_base=20000):
    """
    Assigns unique virtual client IDs to malicious clients.
    Format: 2xyy where:
      - x = real malicious client ID
      - yy = counter for how many times this real client has been used
    """
    virtual_malicious_clients = []
    malicious_client_mapping = dict()     # virtual_id → real_id
    client_region_mapping = dict()        # virtual_id → region_id

    usage_counter = dict()  # real_id → count

    for region_id in regions_to_attack:
        for real_client_id in selected_malicious_clients:
            # Track usage per real client
            if real_client_id not in usage_counter:
                usage_counter[real_client_id] = 0

            count = usage_counter[real_client_id]
            virtual_id = virtual_id_base + real_client_id * 100 + count   # e.g., 20200, 20201, 20300, etc.

            virtual_malicious_clients.append(virtual_id)
            malicious_client_mapping[virtual_id] = real_client_id
            client_region_mapping[virtual_id] = region_id

            usage_counter[real_client_id] += 1

    return virtual_malicious_clients, malicious_client_mapping, client_region_mapping




def get_regions_to_attack(selected_malicious_clients, canonical_client_for_region):
    """
    Given selected malicious clients and the known mapping from region → malicious client,
    determine which regions are being attacked in this round.
    """
    regions_to_attack = set()
    reverse_mapping = {v: k for k, v in canonical_client_for_region.items()}  # client → region

    for client_id in selected_malicious_clients:
        if client_id in reverse_mapping:
            regions_to_attack.add(reverse_mapping[client_id])
        else:
            raise ValueError(f"Malicious client ID {client_id} not found in canonical mapping.")

    return sorted(list(regions_to_attack))


def analyze_malicious_contribution(
    client_weights: dict,
    selected_clients: list,
    selected_malicious_clients: list,
    logger=None
):
    """
    Analyzes how much total weight was assigned to malicious clients
    compared to their proportion in the selected clients.

    Args:
        client_weights (dict): client_id -> weight (percentage)
        selected_clients (list): all selected client IDs this round
        selected_malicious_clients (list): subset of selected_clients that are malicious
        logger (optional): logger instance for logging (can be None)

    Returns:
        dict with:
            - malicious_weight_percent: total assigned weight to malicious clients (0–100%)
            - malicious_client_ratio: fraction of selected clients that are malicious (0–1)
    """
    malicious_weight_percent = sum(
        client_weights.get(cid, 0.0) for cid in selected_malicious_clients
    )

    malicious_client_ratio = len(selected_malicious_clients) / max(len(selected_clients), 1)

    if logger:
        logger.info("[Malicious Contribution Analysis]")
        logger.info(f"  - Malicious clients count: {len(selected_malicious_clients)} / {len(selected_clients)}")
        logger.info(f"  - Malicious client ratio: {malicious_client_ratio:.2f}")
        logger.info(f"  - Total malicious weight: {malicious_weight_percent:.2f}")

    return {
        "malicious_weight_percent": malicious_weight_percent,
        "malicious_client_ratio": malicious_client_ratio
    }



def test_model_asr_acc(
    model,
    test_dataloader,
    device,
    *,
    trigger=None,
    mask=None,
    client_id=None,
    region_id=None,
    loss_fn=None,
    poisoned_batch_injection=None,
    debug_poisonlogger: bool = False,   # <— new
    debug_poison_every: int = 20, # <— new
    logger=None                       # <— new (optional)
) -> dict:
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()

    @torch.no_grad()
    def _eval_once(poisoned: bool):
        total, correct, loss_sum = 0, 0, 0.0
        for bidx, batch in enumerate(test_dataloader):
            if poisoned:
                assert poisoned_batch_injection is not None
                inputs, labels = poisoned_batch_injection(
                    batch=batch, trigger=trigger, mask=mask,
                    is_eval=True, client_id=client_id, region_id=region_id
                )
                if debug_poisonlogger and logger and (bidx % max(1, debug_poison_every) == 0):
                    # lightweight fingerprints
                    flip_rate = float((labels != batch[1].to(labels.device)).float().mean().item())
                    mask_fp   = float(mask.float().sum().item()) if mask is not None else -1.0
                    trig_fp   = float(trigger.float().norm().item()) if trigger is not None else -1.0
                    logger.info(f"[ASR-Eval] region={region_id}, client={client_id}, "
                                f"flip_rate={flip_rate:.3f}, mask_fp={mask_fp:.3f}, trig_fp={trig_fp:.3f}")
            else:
                inputs, labels = batch[0], batch[1]

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)

            pred = outputs.argmax(dim=1)
            correct  += (pred == labels).sum().item()
            total    += labels.size(0)
            loss_sum += loss.item() * labels.size(0)

        acc = correct / max(total, 1)
        avg_loss = loss_sum / max(total, 1)
        return acc, avg_loss

    clean_acc, clean_loss = _eval_once(poisoned=False)
    asr_acc, asr_loss = (None, None)
    if trigger is not None and mask is not None:
        asr_acc, asr_loss = _eval_once(poisoned=True)

    if was_training:
        model.train()

    return {"clean_acc": clean_acc, "clean_loss": clean_loss, "asr": asr_acc, "asr_loss": asr_loss}



def eval_and_log_local(
    *,
    model,
    test_loader,
    device,
    client_id: int,
    region_id: int,
    trigger,
    mask,
    poisoned_batch_injection,
    logger,
    tag: str = "eval",
    show_trigger_dbg: bool = False,
    **test_kwargs,  # passed through to test_model_asr_acc if you add extras later
):
    """
    Run test_model_asr_acc and log a compact line.
    If show_trigger_dbg=True, also print the trigger/mask fingerprints once for this call.
    """
    # — optional "TriggerDbg" line —
    if show_trigger_dbg:
        try:
            trig_fp = _tiny_fp(trigger)
            mask_fp = _tiny_fp(mask)
        except NameError:
            # Fallback if _tiny_fp isn't in scope
            def _fp_trig(t):
                if t is None: return -1.0
                return float(t.float().norm().item())
            def _fp_mask(m):
                if m is None: return -1.0
                return float(m.float().sum().item())
            trig_fp = _fp_trig(trigger)
            mask_fp = _fp_mask(mask)
        logger.info(
            f"[TriggerDbg][Client {client_id} R{region_id}] "
            f"eval trig_fp={trig_fp}, mask_fp={mask_fp}"
        )

    # — run eval —
    metrics = test_model_asr_acc(
        model=model,
        test_dataloader=test_loader,
        device=device,
        trigger=trigger,
        mask=mask,
        client_id=client_id,
        region_id=region_id,
        poisoned_batch_injection=poisoned_batch_injection,
        **test_kwargs,
    )

    asr_pct = (metrics["asr"] * 100.0) if (metrics.get("asr") is not None) else float("nan")
    logger.info(
        f"[LocalEval][Client {client_id}] {tag}: "
        f"clean_acc={metrics['clean_acc']*100:.2f}%, ASR={asr_pct:.2f}%"
    )
    return metrics



def build_pooled_malicious_test_loader(malicious_client, malicious_ids, batch_size):
    """
    Build a 'malicious view' loader by pooling data from the specified malicious clients.

    Preference order:
      1) Pool per-client *train* datasets for the given malicious IDs (fits MSPDataloader)
      2) If (1) not available, pool per-client *test* datasets (if you ever add them)
      3) If neither per-client is available, fall back to the shared test DataLoader (last resort)

    NOTE: Using train datasets means your eval will inherit the *train transforms*
          (e.g., RandomCrop/Flip for CIFAR). That's fine for a quick attacker-view,
          but expect some stochasticity. If you need a strict eval-style transform,
          we can add a transform override later.
    """
    td = getattr(malicious_client, "test_dataloader", None)
    tr = getattr(malicious_client, "train_dataloader", None)

    # --- Preferred path: per-client TRAIN loaders (your MSPDataloader case) ---
    if isinstance(tr, Mapping) or (isinstance(tr, Sequence) and not isinstance(tr, DataLoader)):
        datasets = []
        for cid in malicious_ids:
            try:
                dl = tr[cid]
            except Exception:
                continue
            if isinstance(dl, DataLoader) and hasattr(dl, "dataset"):
                datasets.append(dl.dataset)
        if datasets:
            logger.info("[MaliciousView] Pooled malicious clients' *train* datasets for evaluation.")
            pooled = ConcatDataset(datasets)
            return DataLoader(pooled, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=2, pin_memory=True)

    # --- Secondary path: per-client TEST loaders (if you ever change your loader design) ---
    if isinstance(td, Mapping) or (isinstance(td, Sequence) and not isinstance(td, DataLoader)):
        datasets = []
        for cid in malicious_ids:
            try:
                dl = td[cid]
            except Exception:
                continue
            if isinstance(dl, DataLoader) and hasattr(dl, "dataset"):
                datasets.append(dl.dataset)
        if datasets:
            logger.info("[MaliciousView] Pooled malicious clients' *test* datasets for evaluation.")
            pooled = ConcatDataset(datasets)
            return DataLoader(pooled, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=2, pin_memory=True)

    # --- Fallback: a single shared test loader (server-style eval) ---
    if isinstance(td, DataLoader):
        logger.warning("[MaliciousView] No per-client loaders found; using shared *test* DataLoader as last resort.")
        return td

    # --- Last-last resort: a single shared train loader (very unusual) ---
    if isinstance(tr, DataLoader):
        logger.warning("[MaliciousView] Using shared *train* DataLoader as malicious view (last resort).")
        return tr

    raise TypeError(
        "Cannot build pooled malicious test loader: "
        "no per-client train/test loaders available and no shared test loader found."
    )


# --- NEW: pick a loader by view name -------------------------------------------------
def _pick_loader_for_view(view_name, server_test_loader, mali_test_loader):
    if view_name == "server":
        return server_test_loader
    elif view_name == "malicious":
        return mali_test_loader
    else:
        raise ValueError(f"Unknown view_name: {view_name}")

# --- NEW: dual-view wrapper around test_model_asr_acc --------------------------------
def test_model_asr_acc_two_views(
    *,
    model,
    device,
    server_test_loader=None,
    mali_test_loader=None,
    trigger=None,
    mask=None,
    client_id=None,
    region_id=None,
    poisoned_batch_injection=None,
    logger=None,
    debug_poisonlogger=False,
    debug_poison_every=20,
):
    """
    Returns: {"server": {...}, "malicious": {...}} for whichever loaders are provided.
    Each inner dict has: clean_acc, clean_loss, asr, asr_loss
    """
    out = {}

    def _run(view, loader):
        if loader is None:
            return None
        return test_model_asr_acc(
            model=model,
            test_dataloader=loader,
            device=device,
            trigger=trigger,
            mask=mask,
            client_id=client_id,
            region_id=region_id,
            poisoned_batch_injection=poisoned_batch_injection,
            debug_poisonlogger=debug_poisonlogger,
            debug_poison_every=debug_poison_every,
            logger=logger,
        )

    out["server"]    = _run("server", server_test_loader)
    out["malicious"] = _run("malicious", mali_test_loader)
    return out

# --- NEW: dual-view local eval logger used during crafting ---------------------------
def eval_and_log_local_dual(
    *,
    model,
    device,
    server_test_loader=None,
    mali_test_loader=None,
    client_id: int,
    region_id: int,
    trigger,
    mask,
    poisoned_batch_injection,
    logger,
    tag: str = "eval",
    show_trigger_dbg: bool = False,
    views=("malicious",),  # default: only log malicious view during crafting
    **test_kwargs,
):
    """
    Run dual-view eval and log per requested views.
    Returns {"server": {...} or None, "malicious": {...} or None}
    """
    # Optional fingerprints
    if show_trigger_dbg:
        from .utils import _tiny_fp  # ensure available
        logger.info(
            f"[TriggerDbg][Client {client_id} R{region_id}] "
            f"eval trig_fp={_tiny_fp(trigger)}, mask_fp={_tiny_fp(mask)}"
        )

    res = test_model_asr_acc_two_views(
        model=model,
        device=device,
        server_test_loader=server_test_loader,
        mali_test_loader=mali_test_loader,
        trigger=trigger,
        mask=mask,
        client_id=client_id,
        region_id=region_id,
        poisoned_batch_injection=poisoned_batch_injection,
        logger=logger,
        **test_kwargs,
    )

    for v in views:
        m = res.get(v)
        if m is None:
            continue
        asr_pct = (m["asr"] * 100.0) if (m.get("asr") is not None) else float("nan")
        logger.info(
            f"[{v.capitalize()}LocalEval][Client {client_id}] {tag}: "
            f"clean_acc={m['clean_acc']*100:.2f}%, ASR={asr_pct:.2f}%"
        )
    return res
