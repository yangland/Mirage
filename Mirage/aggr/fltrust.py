# aggr/fltrust.py
import torch
import torch.nn as nn
from torch import linalg as LA
import math
import logging
from torch.utils.data import DataLoader
from copy import deepcopy
from participants.clients.BenignClient import BenignClient
from utils.utils import model_weight_diff, grad_weighted_sum, has_non_finite_tensor
from collections import Counter
import random

logger = logging.getLogger(__name__)

def fltrust_aggr(server_sd, client_grad_dict, **kwargs):
    """
    FLTrust aggregation method.

    Args:
        server_sd: state_dict of the global server model
        client_grad_dict: dict of client_id -> client_update (state_dict)
        kwargs: must include:
            - device
            - root_ds
            - mco_dict
            - iteration
            - batch_size (optional)
            - c_epochs (optional)

    Returns:
        Aggregated update (state_dict)
    """
    device = kwargs["device"]
    root_ds = kwargs["root_ds"]
    iteration = kwargs.get("iteration", 0)
    client_selection = list(client_grad_dict.keys())
    c_epochs = kwargs.get("c_epochs", 1)
    global_model = kwargs.get("global_model")

    # === Step 1: Train server direction on root data
    server_c_grad = fltrust_server_iteration(
        iteration=iteration,
        c_epochs=c_epochs,
        root_ds=root_ds,
        global_model=global_model, 
        params=kwargs.get("params"),
        device=device
    )

    if has_non_finite_tensor(server_c_grad, name="server_c_grad"):
        logger.warning(f"[DEBUG] Iter {iteration} - Non-finite values in server_c_grad")

    # === Step 2: Compute trust scores and clip values
    FLTrustTotalScore = 1e-9
    trust_score_list = []
    clip_value_list = []

    for client_id in client_selection:
        client_grad = client_grad_dict[client_id]

        if has_non_finite_tensor(client_grad, name=f"client {client_id}"):
            logger.warning(f"[DEBUG] Iter {iteration} - Skipping client {client_id} due to non-finite gradients")
            continue

        client_trust_score, client_clipped_value = cosScoreAndClipValue(server_c_grad, client_grad)

        if not math.isfinite(client_trust_score):
            logger.warning(f"[WARNING] Iter {iteration} - Non-finite trust score for client {client_id}")
        if not torch.isfinite(client_clipped_value).all():
            logger.warning(f"[WARNING] Iter {iteration} - Non-finite clipped value for client {client_id}")

        trust_score_list.append(client_trust_score)
        clip_value_list.append(client_clipped_value)
        FLTrustTotalScore += client_trust_score

    # === Step 3: Normalize trust scores
    trust_score_list = [x / FLTrustTotalScore for x in trust_score_list]
    fltrust_weights_list = [a * b for a, b in zip(trust_score_list, clip_value_list)]
    fltrust_weights = dict(zip(client_selection, fltrust_weights_list))

    float_weights = {k: v.item() for k, v in fltrust_weights.items()}
    logger.info(f"[FLTrust] Weights: {float_weights}")

    # === Step 4: Aggregate using weighted sum
    aggregated_grad = grad_weighted_sum(client_grad_dict, fltrust_weights)

    return aggregated_grad



def fltrust_server_iteration(iteration, c_epochs, root_ds, global_model, params, device):
    """
    Trains a copy of the server model on the clean root dataset using BenignClient logic.

    Args:
        iteration (int): Current federated round
        c_epochs (int): Number of local training epochs
        root_ds (Dataset): Trusted IID root dataset
        global_model (nn.Module): The global model before aggregation
        params (dict): Global training params
        device (torch.device): Target device

    Returns:
        state_dict: Difference between trained model and original (i.e., server gradient)
    """
    # === Clone model
    model_copy = deepcopy(global_model).to(device)
    model_copy.train()

    # === Clone params for optimizer, loss, etc.
    root_loader = DataLoader(root_ds, batch_size=params.get("fltrust_batch_size", 16), shuffle=True)

    # === Create dummy benign client
    dummy_client = BenignClient(params=params, train_dataloader=None, test_dataloader=None)

    # === Save model state before training
    model_before = deepcopy(model_copy.state_dict())

    # === Train using BenignClient logic
    dummy_client.local_train(
        iteration=iteration,
        model=model_copy,
        train_loader=root_loader,
        client_id="fltrust_server"
    )

    # === Get model state after training
    model_after = model_copy.state_dict()

    # === Compute diff (gradient)
    grad = model_weight_diff(model_after, model_before)

    return grad



def modelsd2tensor(model_dict):
    ravel_list = []
    for layer_name, parms in model_dict.items():
        ravel_list.append(torch.ravel(parms))
    ravel_list = torch.cat(ravel_list, 0)
    return torch.unsqueeze(ravel_list, 0)


def relu_cos(a, b):
    # a, b as tensors 
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos = cos_sim(a, b)
    
    '''relu'''
    if cos < 0:
        cos = 0
    return cos


def norm_clip(a, b):
    # a, b as tensors
    res = LA.norm(a)/(LA.norm(b)+1e-9)
    return res


def cosScoreAndClipValue(model_dict1, model_dict2):
    t1 = modelsd2tensor(model_dict1)
    t2 = modelsd2tensor(model_dict2)
    return relu_cos(t1, t2), norm_clip(t1, t2)


def create_fl_trust_root_dataset(fl_train, per_class_sample):


    # Extract all targets
    train_targets = [x[1] for x in fl_train]

    # Get all unique class labels
    all_labels = sorted(set(train_targets))
    fl_class_num = len(all_labels)

    # Sample fixed number of examples per class
    indices = []
    for label in all_labels:
        label_indices = [idx for idx, target in enumerate(train_targets) if target == label]
        if len(label_indices) < per_class_sample:
            raise ValueError(f"Not enough samples for class {label} (needed {per_class_sample}, found {len(label_indices)})")
        sampled = random.sample(label_indices, per_class_sample)
        indices += sampled

    # Create subset
    ds = torch.utils.data.Subset(fl_train, indices)

    # Print label distribution
    label_counts = Counter([fl_train[i][1] for i in indices])
    print("Label distribution in FLTrust root dataset:")
    for label in sorted(label_counts):
        print(f"  Class {label}: {label_counts[label]} samples")

    return ds
