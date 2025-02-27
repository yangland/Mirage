import copy
import logging
import random
import numpy as np

import torch
import torch.nn as nn

import torchvision.models
import models.resnet
import models.vgg
from utils.utils import poisoned_batch_injection
from utils.visualize import visualize, visualize_batch, visualize_tsne

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

    def test_global_model(self, iteration, malicious_clients):
        '''
        clean acc 和 asr
        :param iteration: current iteration
        :param malicious_clients: attacker
        :return: acc -> float, acc_loss -> float, asr_list -> [float, float, float], asr_loss -> [float, float, float]
        '''
        acc, acc_loss = self.test_model_once(iteration, self.test_dataloader, is_poisoned=False)
        logger.info(f"{'-----------------------------------------------':<50}")
        logger.info(f"|Test Global model in iteration {iteration}")
        logger.info(f"|Loss {acc_loss:.4f}, Acc {acc * 100:.2f}%")

        if iteration >= self.params["poisoned_start_iteration"]:
            for attacker_id in range(self.params["no_of_adversaries"]):
                trigger = malicious_clients.trigger_set[attacker_id]
                mask = malicious_clients.mask_set[attacker_id]
                asr, loss = self.test_model_once(iteration, self.test_dataloader, is_poisoned=True,
                                                 trigger=trigger, mask=mask,
                                                 label_swap=self.params["poison_label_swap"][attacker_id])
                self.acc_p_list[attacker_id].append(asr)
                logger.info(f"| Attacker {attacker_id}: Loss {loss:.4f}, ASR {asr * 100:.2f}%")

        logger.info(f"{'-----------------------------------------------':<50}")
        self.acc_list.append(acc)
        if iteration % 50 == 0:
            logger.info(f"acc_list: {self.acc_list}")
            for i in range(self.params["no_of_adversaries"]):
                logger.info(f"ASR of attacker {i}: {self.acc_p_list[i]}")

        # TODO plot t-sne
        show_tsne = False
        if show_tsne and (iteration % 10 == 0 or (
                self.params["malicious_train_algo"] == "Mirage" and iteration % 3 == 0 and iteration - self.params[
            "start_save_iteration"] <= 101)):
            cache_samples = torch.tensor([])
            cache_labels = torch.tensor([])
            # 干净分布
            copy_model = copy.deepcopy(self.global_model).to(torch.device("cpu"))
            copy_model.eval()
            copy_model.linear = torch.nn.Sequential()
            for ind, (samples, labels) in malicious_clients.test_sample_cache.items():
                cache_samples = torch.cat((cache_samples, samples), dim=0)
                cache_labels = torch.cat((cache_labels, labels), dim=0)
            cache_features = copy_model(cache_samples)
            indices = torch.randperm(len(cache_samples))
            for client_id in range(self.params["no_of_adversaries"]):
                samples = cache_samples[indices][200 * client_id:200 * (client_id + 1)]
                labels = cache_labels[indices][200 * client_id:200 * (client_id + 1)]
                poi_inputs, poi_labels = poisoned_batch_injection((samples, labels),
                                                                  trigger=malicious_clients.trigger_set[client_id],
                                                                  mask=malicious_clients.mask_set[client_id],
                                                                  is_eval=True,
                                                                  label_swap=self.params["poison_label_swap"][
                                                                      client_id])

                poi_inputs = poi_inputs.to(torch.device("cpu"))
                poi_labels = poi_labels.to(torch.device("cpu"))

                features_poi = copy_model(poi_inputs)
                cache_features = torch.cat((cache_features, features_poi), dim=0)
                cache_labels = torch.cat((cache_labels, poi_labels + 100), dim=0)

            visualize_tsne(cache_features, cache_labels, iteration, self.params["folder_path"])
        return

    def test_model_once(self, iteration, test_dataloader, is_poisoned=False, model=None, trigger=None, mask=None,
                        label_swap=0):
        '''
        test model
        :param iteration: current iterations
        :param test_dataloader:  test dataloader
        :param is_poisoned: is poison
        :param trigger: trigger for testing, shape, (channel, height, width)
        :param mask: trigger mask, shape (channel, height, width)
        :param label_swap: labels
        :return: results，acc, loss
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
                    # 如果需要测试ASR，则需要对batch进行投毒
                    sample_indices = ~(batch[1] == label_swap)
                    samples = batch[0][sample_indices]
                    labels = batch[1][sample_indices]
                    batch = poisoned_batch_injection((samples, labels), trigger, mask, is_eval=True,
                                                     label_swap=label_swap)
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
            ### don't scale tied weights:
            if name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def create_global_model_copy(self):
        global_model_copy = dict()
        for name, param in self.global_model.named_parameters():
            global_model_copy[name] = self.global_model.state_dict()[name].clone().detach().requires_grad_(False)
        return global_model_copy

    def aggregation(self, weight_accumulator, aggregated_model_id):
        '''
        model aggregation

        :param weight_accumulator:
        :param aggregated_model_id:
        :param update_norm_list:
        :return:
        '''

        no_of_participants_this_round = sum(aggregated_model_id)
        if no_of_participants_this_round != 0:
            for name, data in self.global_model.state_dict().items():
                update_per_layer = weight_accumulator[name] * (1 / no_of_participants_this_round)
                data = data.float()
                data.add_(update_per_layer)
