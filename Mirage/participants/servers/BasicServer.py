import copy
import logging
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models
from torch.nn import functional as F

import models.resnet
import models.vgg
from utils.utils import poisoned_batch_injection
from utils.visualize import visualize, visualize_batch, visualize_tsne
from aggr import aggregate_global_model


logger = logging.getLogger("logger")


class \
        BasicServer():
    def __init__(self, params, dataloader):
        self.params = params
        self.train_dataloader = dataloader.train_dataloader
        self.test_dataloader = dataloader.test_dataloader

        self.acc_list = list()
        self.acc_p_list = [list() for i in range(self.params["no_of_adversaries"])]
        self.model = None
        self.create_model()
        self.resume_model()
        self.poisoned_iterations = [iteration for iteration in range(self.params["poisoned_start_iteration"],
                                                                     self.params["poisoned_end_iteration"],
                                                                     self.params["poisoned_iteration_interval"])]
        self.test_model_once(-1, self.test_dataloader, is_poisoned=False)
        self.region_index = 0

    def pre_process(self, *args, **kwargs):

        return True

    def create_model(self):
        print(f"creating model")
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                global_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                global_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "GTSRB":
                global_model = getattr(models.resnet, self.params["model_type"])(num_classes=43, dataset="GTSRB")
            elif self.params["dataset"].upper() == "EMNIST":
                global_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")

        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                global_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                global_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)
        elif "MobileNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                global_model = torchvision.models.mobilenet_v2(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                global_model = torchvision.models.mobilenet_v2(num_classes=100)

        self.global_model = global_model.to(self.params["run_device"])

        return True

    def resume_model(self):
        if self.params["resumed_model"]:
            loads = torch.load(f"{self.params['resumed_model']}", map_location=self.params["run_device"])
            if type(loads) == dict and "state_dict" in loads.keys():

                self.global_model.load_state_dict(loads["state_dict"])

                self.params["start_iteration"] = loads["iteration"]
                logger.info(
                    f"Loaded params from saved model, LR is {self.params['benign_lr']} and current iteration is {self.params['start_iteration']}")
            elif type(loads) == dict and "model" in loads.keys():
                self.global_model.load_state_dict(loads["model"])
                logger.info(
                    f"Loaded params from saved model, LR is {self.params['benign_lr']} and current iteration is {self.params['start_iteration']}")
        else:
            self.params["start_iteration"] = 1
            logger.info(f"start training from the 1st round")
            
    def select_clients(self, iteration):
        r"""
        randomly select participating clients for each round
        """
        if "poison_type" not in self.params.keys():
            self.params["poison_type"] = "continue_poison"

        adversary_list = []
        if iteration in self.poisoned_iterations:
            if 'continue_poison' in self.params["poison_type"]:
                adversary_list = [i for i in range(self.params["no_of_adversaries"])]
                selected_clients = adversary_list + random.sample(
                    range(len(adversary_list), self.params["no_of_total_participants"]),
                    self.params["no_of_participants_per_iteration"] - len(adversary_list))

            elif 'full_random' in self.params["poison_type"]:
                selected_clients = random.sample(
                    range(self.params["no_of_total_participants"]),
                    self.params["no_of_participants_per_iteration"] - len(adversary_list))
                adversary_list = [i for i in selected_clients if i < self.params["no_of_adversaries"]]
            elif 'sequential_poison' in self.params["poison_type"]:
                adversary_list = [iteration % self.params["no_of_adversaries"]]
                selected_clients = adversary_list + random.sample(
                    range(len(adversary_list), self.params["no_of_total_participants"]),
                    self.params["no_of_participants_per_iteration"] - len(adversary_list))
        else:
            selected_clients = random.sample(
                range(self.params["no_of_total_participants"]),
                self.params["no_of_participants_per_iteration"] - len(adversary_list))

        logger.info(f"selected clients: {selected_clients}")
        logger.info(f"adversary list: {adversary_list}")

        return selected_clients, adversary_list


    # def test_global_model(self, iteration, malicious_clients):
    #     """
    #     Evaluate the global model: clean accuracy and ASR for each attacker.
    #     Returns a dictionary with clean accuracy/loss and ASR info.
    #     """
    #     results = {
    #         "clean_acc": None,
    #         "clean_loss": None,
    #         "asr": {}
    #     }

    #     # === Clean evaluation
    #     acc, acc_loss = self.test_model_once(iteration, self.test_dataloader, is_poisoned=False)
    #     self.acc_list.append(acc)
    #     results["clean_acc"] = acc
    #     results["clean_loss"] = acc_loss

    #     logger.info(f"{'-'*55}")
    #     logger.info(f"| Test Global model in iteration {iteration}")
    #     logger.info(f"| Loss {acc_loss:.4f}, Acc {acc * 100:.2f}%")

    #     # === ASR evaluation (if poisoned round)
    #     if iteration >= self.params["poisoned_start_iteration"]:
    #         for attacker_id in range(self.params["no_of_adversaries"]):
    #             trigger = malicious_clients.trigger_set[attacker_id]
    #             mask = malicious_clients.mask_set[attacker_id]
    #             label_swap = self.params["poison_label_swap"][attacker_id]

    #             asr, loss = self.test_model_once(
    #                 iteration,
    #                 self.test_dataloader,
    #                 is_poisoned=True,
    #                 trigger=trigger,
    #                 mask=mask,
    #                 label_swap=label_swap
    #             )

    #             self.acc_p_list[attacker_id].append(asr)
    #             logger.info(f"| Attacker {attacker_id}: Loss {loss:.4f}, ASR {asr * 100:.2f}%")

    #             results["asr"][attacker_id] = {
    #                 "asr": asr,
    #                 "loss": loss
    #             }

    #     logger.info(f"{'-'*55}")

    #     if iteration % 50 == 0:
    #         logger.info(f"acc_list: {self.acc_list}")
    #         for i in range(self.params["no_of_adversaries"]):
    #             logger.info(f"ASR of attacker {i}: {self.acc_p_list[i]}")

    #     # Optional: TSNE visualization
    #     show_tsne = False
    #     if show_tsne and (iteration % 10 == 0 or (
    #             self.params["malicious_train_algo"] == "Mirage"
    #             and iteration % 3 == 0
    #             and iteration - self.params["start_save_iteration"] <= 101)):
    #         self._visualize_tsne(iteration, malicious_clients)

    #     return results


    # def test_model_once(self, iteration, test_dataloader, is_poisoned=False, model=None, trigger=None, mask=None,
    #                     label_swap=0):
    #     '''
    #     test model
    #     :param iteration: current iterations
    #     :param test_dataloader:  test dataloader
    #     :param is_poisoned: is poison
    #     :param trigger: trigger for testing, shape, (channel, height, width)
    #     :param mask: trigger mask, shape (channel, height, width)
    #     :param label_swap: labels
    #     :return: results，acc, loss
    #     '''
    #     if model is None:
    #         model = copy.deepcopy(self.global_model)
    #     model.eval()
    #     with torch.no_grad():
    #         total_loss = 0.
    #         total_correct = 0.
    #         total_num = 0.

    #         criterion = nn.CrossEntropyLoss(reduction='sum')
    #         for i, batch in enumerate(test_dataloader):
    #             if is_poisoned:
    #                 # 如果需要测试ASR，则需要对batch进行投毒
    #                 sample_indices = ~(batch[1] == label_swap)
    #                 samples = batch[0][sample_indices]
    #                 labels = batch[1][sample_indices]
    #                 batch = poisoned_batch_injection(batch=(samples, labels), 
    #                                                  trigger=trigger, 
    #                                                  mask=mask, 
    #                                                  is_eval=True,
    #                                                  client_id=None,  # No client ID needed for ASR testing
    #                                                  region_mapping=None)  # TODO, this function is not been used
    #             data, target = batch
    #             data, target = data.to(self.params["run_device"]), target.to(self.params["run_device"])
    #             output = model(data)
    #             loss = criterion(output, target)
    #             total_correct += (output.argmax(dim=1) == target).sum().item()
    #             total_loss += loss.item()
    #             total_num += data.size(0)
    #     acc = total_correct / total_num
    #     loss = total_loss / total_num
    #     return acc, loss


    def test_global_model(self, iteration, malicious_clients, possible_region_ids=None, client_region_mapping=None, show_tsne=False):
        """
        Evaluate the global model: clean accuracy and ASR per region.
        Returns a dictionary with clean accuracy/loss and ASR info.
        """
        results = {
            "clean_acc": None,
            "clean_loss": None,
            "asr": {}
        }

        # === Clean evaluation ===
        acc, acc_loss = self.test_model_once(iteration, self.test_dataloader, is_poisoned=False)
        self.acc_list.append(acc)
        results["clean_acc"] = acc
        results["clean_loss"] = acc_loss

        logger.info(f"{'-'*55}")
        logger.info(f"| Test Global model in iteration {iteration}")
        logger.info(f"| Loss {acc_loss:.4f}, Acc {acc * 100:.2f}%")

        # === ASR evaluation per region ===
        if iteration >= self.params["poisoned_start_iteration"]:
            for region_id in possible_region_ids:
                if region_id not in malicious_clients.trigger_set_by_region or \
                region_id not in malicious_clients.mask_set_by_region:
                    logger.warning(f"[ASR] Skipping Region {region_id} — no trigger/mask available.")
                    continue

                trigger = malicious_clients.trigger_set_by_region[region_id]
                mask = malicious_clients.mask_set_by_region[region_id]

                # Pick any available client from this region to perform ASR test
                test_client_ids = [cid for cid, rid in client_region_mapping.items() if rid == region_id]
                if not test_client_ids:
                    logger.warning(f"[ASR] Skipping Region {region_id} — no clients mapped for testing.")
                    continue

                test_client_id = test_client_ids[0]  # Pick any one client

                asr, loss = self.test_model_once(
                    iteration=iteration,
                    test_dataloader=self.test_dataloader,
                    is_poisoned=True,
                    trigger=trigger,
                    mask=mask,
                    client_id=test_client_id,
                    client_region_mapping=client_region_mapping
                )

                logger.info(f"| Region {region_id}: ASR {asr * 100:.2f}%, Loss {loss:.4f}")
                results["asr"][region_id] = {
                    "asr": asr,
                    "loss": loss
                }

        logger.info(f"{'-'*55}")

        if iteration % 50 == 0:
            logger.info(f"acc_list: {self.acc_list}")
            for region_id in possible_region_ids:
                if region_id in results["asr"]:
                    logger.info(f"ASR of region {region_id}: {results['asr'][region_id]['asr'] * 100:.2f}%")

        # Optional: TSNE visualization
        if show_tsne and (iteration % 10 == 0 or (
                self.params["malicious_train_algo"] == "Mirage"
                and iteration % 3 == 0
                and iteration - self.params["start_save_iteration"] <= 101)):
            self._visualize_tsne(iteration, malicious_clients)

        return results


    def test_model_once(self, iteration, test_dataloader, is_poisoned=False, model=None, trigger=None, mask=None,
                        client_id=None, client_region_mapping=None):
        '''
        Test model once (clean or poisoned).
        '''
        if model is None:
            model = copy.deepcopy(self.global_model)
        model.eval()

        with torch.no_grad():
            total_loss = 0.
            total_correct = 0.
            total_num = 0.

            criterion = nn.CrossEntropyLoss(reduction='sum')
            for i, batch in enumerate(test_dataloader):
                if is_poisoned:
                    if client_id is None or client_region_mapping is None:
                        raise ValueError("client_id and region_mapping must be provided for poisoned testing.")

                    # Get region ID and label swap
                    region_id = client_region_mapping[client_id]
                    label_swap = self.params["poison_label_swap_by_region"][region_id]

                    # Escape clean samples of target class
                    sample_indices = ~(batch[1] == label_swap)
                    samples = batch[0][sample_indices]
                    labels = batch[1][sample_indices]

                    batch = poisoned_batch_injection(
                        batch=(samples, labels),
                        trigger=trigger,
                        mask=mask,
                        is_eval=True,
                        client_id=client_id,
                        region_id=region_id
                    )

                data, target = batch
                data, target = data.to(self.params["run_device"]), target.to(self.params["run_device"])
                output = model(data)
                loss = criterion(output, target)
                total_correct += (output.argmax(dim=1) == target).sum().item()
                total_loss += loss.item()
                total_num += data.size(0)

        acc = total_correct / total_num
        loss = total_loss / total_num
        return acc, loss


    def save_model(self, iteration, trigger_set, mask_set):
        trigger_set = copy.deepcopy(trigger_set)
        mask_set = copy.deepcopy(mask_set)
        avg_ASR = np.mean(np.array(self.acc_p_list), axis=0)

        save_flag = False
        if iteration in self.params["save_on_iteration"]:
            print(f"save model on milestone iteration {iteration}")
            file_name = f"{self.params['model_type']}_{iteration}"
            save_flag = True
        elif len(avg_ASR) > 2:
            if iteration > self.params["start_save_iteration"] and (
                    round(avg_ASR[-1], 4) == round(max(avg_ASR), 4)) and (
                    np.count_nonzero(avg_ASR == np.max(avg_ASR)) <= 4):
                print(f"save model with best ASR on iteration {iteration}")
                file_name = f"Best_ASR_{self.params['model_type']}_{iteration}"
                save_flag = True
        elif iteration == self.params["end_iteration"] - 1:
            print(f"save model on end iteration {iteration}")
            file_name = f"{self.params['model_type']}_{iteration}_end_model"
            save_flag = True
        if save_flag:
            save_flag = False
            logger.info(f"saving model on iteration {iteration}")
            for i in range(len(trigger_set)):
                trigger_set[i] = trigger_set[i].cpu().numpy()
                mask_set[i] = mask_set[i].cpu().numpy()
            file_name = f"{self.params['folder_path']}/{file_name}.pt.tar"
            self.global_model.to(torch.device("cpu"))
            saved_dict = {
                "state_dict": self.global_model.state_dict(),
                "iteration": iteration,
                "trigger_set": trigger_set,
                "mask_set": mask_set,
                "acc_list": self.acc_list,
                "acc_p_list": self.acc_p_list,
                # "ood_dataloader": ood_dataloader
            }
            torch.save(saved_dict, file_name)
            logger.info(f"model saved to {file_name}")

            self.global_model.to(self.params["run_device"])

    def create_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.global_model.state_dict().items():
            ### don't scale tied weights, Now use the full state_dict
            # if name == 'decoder.weight' or '__' in name:
            #     continue
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def create_global_model_copy(self):
        global_model_copy = dict()
        for name, param in self.global_model.named_parameters():
            global_model_copy[name] = self.global_model.state_dict()[name].clone().detach().requires_grad_(False)
        return global_model_copy

    # def aggregation(self, weight_accumulator, aggregated_model_id):
    #     '''
    #     model aggregation

    #     :param weight_accumulator:
    #     :param aggregated_model_id:
    #     :param update_norm_list:
    #     :return:
    #     '''

    #     no_of_participants_this_round = sum(aggregated_model_id)
    #     if no_of_participants_this_round != 0:
    #         for name, data in self.global_model.state_dict().items():
    #             update_per_layer = weight_accumulator[name] * (1 / no_of_participants_this_round)
    #             data = data.float()
    #             data.add_(update_per_layer)

    def aggregation(self, agg_method, weight_accumulator_by_client):
        """
        Aggregates model using specified strategy (FedAvg, Krum, etc.)

        :param weight_accumulator: unused in new version
        :param aggregated_model_id: used to know how many participated
        :param weight_accumulator_by_client: list of state_dicts (updates)
        """
        agg_method = self.params.get("agg_method", "unknown").lower()
        
        # === Convert list of state_dicts to dict[str, state_dict]
        client_grad_dict = {
            f"client_{client_id}": update
            for client_id, update in weight_accumulator_by_client.items()
        }

        # Validate all client updates before aggregation
        for client_id, update in client_grad_dict.items():
            if not isinstance(update, dict):
                print(f"[ERROR] Client {client_id} update is not a dict: {type(update)}")
                print("update:", update)
                raise TypeError(f"[ERROR] Invalid update format for client {client_id}")
            for k, v in update.items():
                if not isinstance(v, torch.Tensor):
                    print(f"[ERROR] Client {client_id} update param {k} is not a Tensor: {type(v)}")
                    raise TypeError(f"[ERROR] Invalid param in update for client {client_id}")

        # === Determine additional params for Krum-based methods
        extra_args = {}
        num_clients = len(client_grad_dict)

        if agg_method in ["krum"]:
            f = num_clients // 2 - 1  # Byzantine tolerance
            m = num_clients - f - 2   # For multi-krum, number of selected updates
            extra_args["f"] = f
            extra_args["m"] = m

            print(f"[INFO] Using {agg_method} with f={f}" + (f", m={m}" if agg_method == "multi-krum" else ""))

        # === Call aggregation dispatcher
        aggregated_update = aggregate_global_model(
            agg_method=agg_method,
            server_model=self.global_model,
            client_grad_dict=client_grad_dict,
            params=self.params,
            iteration=getattr(self, "current_iteration", 0),
            **extra_args,  # Pass extra krum-specific args
        )


        # === Apply aggregated update to global model
        for name, param in self.global_model.state_dict().items():
            # param.add_(aggregated_update[name].to(param.device))
            update_tensor = aggregated_update[name].to(dtype=param.dtype, device=param.device)
            param.add_(update_tensor)


    def _visualize_tsne(self, iteration, malicious_clients):
        """
        Visualize feature embeddings using t-SNE for selected clients.
        """
        logger.info(f"[TSNE] Visualizing feature embeddings at iteration {iteration}...")

        device = next(self.model.parameters()).device

        # === Collect features and labels from all or selected clients ===
        # For simplicity, use the test set (or a subset of it)
        features, labels = extract_features(self.model, self.test_dataloader, device)

        # === Optionally modify or tag labels to distinguish client types ===
        # Example: Label malicious clients with 100+region_id to distinguish
        if hasattr(malicious_clients, "client_region_mapping"):
            for region_id in malicious_clients.trigger_set_by_region.keys():
                indices = (labels == region_id)
                labels[indices] = 100 + region_id  # Mark as malicious for color coding

        # === Create save directory if needed ===
        save_path = os.path.join(self.params.get("folder_path", "."), "tsne")
        os.makedirs(save_path, exist_ok=True)

        # === Call visualization function ===
        visualize_tsne(
            features=features,
            labels_tensor=labels,
            attaches=f"iter{iteration}",
            save_path=save_path
        )

        logger.info(f"[TSNE] Saved t-SNE plot at iteration {iteration} to {save_path}")


def extract_features(model, dataloader, device):
    """
    Helper function to extract features and labels from a model and dataloader.
    Assumes model returns feature vectors from penultimate layer.
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            output = model.extract_features(x) if hasattr(model, "extract_features") else model(x)

            features.append(output)
            labels.append(y)

            # Optional: limit number of batches to speed up tsne
            if len(features) >= 5:  # limit to 5 batches
                break

    return torch.cat(features), torch.cat(labels)
    
    



