import copy
import torch
import logging
import time

from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from tqdm import tqdm
import ot
from participants.clients.BasicClient import BasicClient
from utils.utils import poisoned_batch_injection
from utils.visualize import visualize, visualize_batch, visualize_tsne
from utils.regoin_utils import compute_benign_statistics, is_within_l2_ball, is_within_update_cone, is_within_weight_cone, flatten_model, unflatten_model
import torch.nn.functional as F

logger = logging.getLogger("logger")


class MirageClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader):
        super(MirageClient, self).__init__(params, train_dataloader, test_dataloader)
        self.init_trigger_mask()
        self.asr_before_upload = {}

    def generate_discriminator_dataloader(self, model, train_loader, trigger_, mask_, client_id):
        '''
        discriminator trainset, target class is 0, target class is 1
        :param model:
        :param train_loader:
        :param trigger_:
        :param mask_:
        :param client_id:
        '''

        class_num = self.params["class_num"]
        samples_per_class = {i: torch.tensor([], device=self.params["run_device"]) for i in range(class_num)}
        criterion = nn.CrossEntropyLoss(reduction='none').to(self.params["run_device"])
        label_list = [0 for _ in range(class_num)]
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.params["run_device"]), labels.to(self.params["run_device"])

            for class_ind in range(class_num):
                indices = labels == class_ind
                label_list[class_ind] += sum(indices)
                samples_per_class[class_ind] = torch.cat((samples_per_class[class_ind], inputs[indices]), dim=0)

        target_class = self.params["poison_label_swap"][client_id]
        for i in range(class_num):
            sample = samples_per_class[i]
            if len(sample) == 0:
                continue
            outputs = model(sample)
            tmp_label = torch.ones(len(outputs), dtype=torch.long, device=self.params["run_device"]) * i
            loss_sort_by_samples = criterion(outputs, tmp_label)
            samples_selected_len = self.params["discriminator_train_samples_pre_class"] if len(outputs) > self.params[
                "discriminator_train_samples_pre_class"] else len(outputs)
            if i == target_class:
                samples_selected_len = len(outputs)
            _, indices = torch.topk(loss_sort_by_samples, samples_selected_len,
                                    largest=False)
            representative_samples = sample[indices]
            samples_per_class[i] = representative_samples
        samples_discriminator_dataloader = torch.tensor([], device=self.params["run_device"])
        labels_discriminator_dataloader = torch.tensor([], dtype=torch.long, device=self.params["run_device"])
        for i in range(class_num):
            if i == target_class:
                continue
            samples = samples_per_class[i]
            labels = torch.ones(len(samples), dtype=torch.long, device=self.params["run_device"])
            poisoned_sample, _ = poisoned_batch_injection((samples, labels), trigger=trigger_, mask=mask_, is_eval=True,
                                                          label_swap=target_class)
            samples_discriminator_dataloader = torch.cat((samples_discriminator_dataloader, poisoned_sample), dim=0)
            labels_discriminator_dataloader = torch.cat((labels_discriminator_dataloader, labels), dim=0)

        samples_discriminator_dataloader = torch.cat(
            (samples_discriminator_dataloader, samples_per_class[target_class]), dim=0)
        labels_discriminator_dataloader = torch.cat((labels_discriminator_dataloader,
                                                     torch.zeros(len(samples_per_class[target_class]), dtype=torch.long,
                                                                 device=self.params["run_device"])), dim=0)
        discriminator_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(samples_discriminator_dataloader, labels_discriminator_dataloader),
            batch_size=self.params["discriminator_batch_size"], shuffle=True)

        return discriminator_dataloader

    def get_discriminator(self, model, discriminator_dataloader):
        discriminator_ = copy.deepcopy(model)
        if "resnet" in self.params["model_type"].lower():
            discriminator_.linear = torch.nn.Sequential(
            torch.nn.Linear(discriminator_.linear.in_features, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 2)
        )
        elif "vgg" in self.params["model_type"].lower():
            discriminator_.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.classifier.in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "mobilenet" in self.params["model_type"].lower():
            discriminator_.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.classifier[1].in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )

        for name, param in discriminator_.named_parameters():
            if self.classifier_name not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        discriminator_optimizer = torch.optim.SGD(discriminator_.parameters(), lr=self.params["discriminator_lr"],
                                                  momentum=self.params['discriminator_momentum'],
                                                  weight_decay=self.params['discriminator_weight_decay'])

        discriminator_criterion = nn.CrossEntropyLoss().to(self.params["run_device"])

        discriminator_ = discriminator_.to(self.params["run_device"])

        for iter in range(self.params["discriminator_train_no_times"]):
            total_loss = 0.
            for batch in discriminator_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.params["run_device"]), labels.to(self.params["run_device"])
                outputs = discriminator_(inputs)
                loss = discriminator_criterion(outputs, labels)
                discriminator_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                total_loss += loss.item()
                discriminator_optimizer.step()
        discriminator_.eval()

        return discriminator_

    def search_trigger(self, model, train_loader, client_id, test_loader=None):
        '''
        optimize trigger

        :param model:
        :param train_loader:
        :param client_id:
        :return:
        '''
        model.eval()
        local_train_loader = copy.deepcopy(train_loader)
        trigger_ = copy.deepcopy(self.trigger_set[client_id])
        mask_ = copy.deepcopy(self.mask_set[client_id])
        ce_loss = nn.functional.cross_entropy
        cos_loss = nn.CosineSimilarity(dim = 1,  eps=1e-08)

        feature_extractor = copy.deepcopy(model)
        feature_extractor.linear = torch.nn.Sequential()

        t = copy.deepcopy(trigger_)
        for iters in tqdm(range(self.params["trigger_search_no_times"])):
            dataloader_discriminator = self.generate_discriminator_dataloader(model, local_train_loader, t, mask_,
                                                                              client_id)
            total_loss = 0.
            trigger_optim = torch.optim.Adam([t], lr=self.params["trigger_lr"], weight_decay=5e-4)
            counter = 0
            loss_adv = 0.
            loss_acc = 0.
            model_discriminator = self.get_discriminator(model, dataloader_discriminator)

            for inputs, targets in train_loader:  # 在训练集上更新样本
                t.requires_grad_()
                inputs, targets = inputs.to(self.params["run_device"]), targets.to(self.params["run_device"])
                batch_clean_indices = targets == self.params["poison_label_swap"][client_id]
                if batch_clean_indices.sum() == 0:
                    continue
                counter += 1

                batch_backdoor_indices = ~batch_clean_indices
                backdoor_inputs = inputs[batch_backdoor_indices]
                backdoor_targets = targets[batch_backdoor_indices]

                backdoor_inputs, backdoor_targets = poisoned_batch_injection((backdoor_inputs, backdoor_targets),
                                                                             trigger=t, mask=mask_, is_eval=False,
                                                                             label_swap=
                                                                             self.params["poison_label_swap"][
                                                                                 client_id])

                backdoor_inputs = backdoor_inputs.to(self.params["run_device"])

                # TODO 1 -> to ID,
                backdoor_pred_disc = model_discriminator(backdoor_inputs)
                loss_discriminator = ce_loss(backdoor_pred_disc,
                                             torch.zeros(len(backdoor_pred_disc), device=self.params["run_device"]).long())
                backdoor_pred = model(backdoor_inputs)
                # TODO 2 -> enhancement
                loss_asr = ce_loss(backdoor_pred, backdoor_targets)
                loss_sim = cos_loss(backdoor_pred, model(inputs[batch_backdoor_indices])).mean()

                loss = 0.
                loss += loss_discriminator
                loss += loss_asr
                loss += loss_sim
                total_loss += loss.item()
                if loss != None and loss.item() != 0.:
                    trigger_optim.zero_grad()
                    loss.backward(retain_graph=True)
                    new_t = t - t.grad.sign() * self.params["trigger_lr"]
                    t = new_t.detach()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        return t.detach_()

    def local_train_mirage(self, iteration, model, train_loader, client_id, test_loader=None):
        '''
        poisoning training process

        :param iteration:
        :param model:
        :param train_loader:
        :param client_id:
        :return:
        '''

        cache_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(cache_model.parameters(), lr=self.params['poisoned_lr'],
                                    momentum=self.params['poisoned_momentum'],
                                    weight_decay=self.params['poisoned_weight_decay'])

        trigger_ = self.search_trigger(cache_model, train_loader, client_id)
        self.trigger_set[client_id] = trigger_

        model.train()
        cache_model.train()
        current_time = time.time()

        for epoch in range(self.params["poisoned_retrain_no_times"]):

            cache_model.train()
            total_loss = 0.
            counter = 0.
            for batch_idx, batch in enumerate(train_loader):
                counter += 1
                inputs, labels = poisoned_batch_injection(batch, trigger=self.trigger_set[client_id],
                                                          mask=self.mask_set[client_id], is_eval=False,
                                                          label_swap=self.params["poison_label_swap"][client_id])

                inputs, labels = inputs.to(self.params["run_device"]), labels.to(self.params["run_device"])
                outputs = cache_model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            cache_model.eval()
            # acc, loss = self.local_test_once(cache_model, self.test_dataloader, is_poisoned=False, client_id=client_id)
            # print(f"acc - {acc}, loss - {loss}")
            # asr, loss = self.local_test_once(cache_model, self.test_dataloader, is_poisoned=True, client_id=client_id)
            # print(f"asr - {asr}, loss - {loss}")
        return cache_model


    def local_train(self, iteration, model, train_loader, client_id, test_loader=None, region_constraint=None):
        """
        Modified local training with region-constrained backdoor crafting (PGD).
        Also computes ASR before upload.
        """
        global_model = copy.deepcopy(model)
        cache_model = copy.deepcopy(model)
        cache_model.train()

        optimizer = torch.optim.SGD(cache_model.parameters(), lr=self.params['poisoned_lr'],
                                    momentum=self.params['poisoned_momentum'],
                                    weight_decay=self.params['poisoned_weight_decay'])

        # === Step 1: Optimize the trigger ===
        trigger_ = self.search_trigger(cache_model, train_loader, client_id)
        self.trigger_set[client_id] = trigger_

        mask_ = self.mask_set[client_id]
        target_label = self.params["poison_label_swap"][client_id]
        ce_loss = nn.CrossEntropyLoss().to(self.params["run_device"])

        # === Step 2: Train local model with trigger & region constraint ===
        for epoch in range(self.params["poisoned_retrain_no_times"]):
            for batch_idx, batch in enumerate(train_loader):
                inputs, labels = poisoned_batch_injection(
                    batch,
                    trigger=self.trigger_set[client_id],
                    mask=self.mask_set[client_id],
                    is_eval=False,
                    label_swap=target_label
                )

                inputs, labels = inputs.to(self.params["run_device"]), labels.to(self.params["run_device"])

                outputs = cache_model(inputs)
                loss = ce_loss(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # === Step 3: Project model back into region ℛ_j ===
                delta = flatten_model(cache_model) - flatten_model(global_model)

                # Get projection direction v_j and constraints
                v_j = region_constraint.get("direction")  # axis vector
                tau_j = region_constraint.get("cosine_threshold")
                r_min = region_constraint.get("r_min")
                r_max = region_constraint.get("r_max")

                # Project Δ onto norm bounds
                norm = torch.norm(delta, p=2)
                if r_max is not None and norm > r_max:
                    delta = delta / norm * r_max
                elif r_min is not None and norm < r_min:
                    delta = delta / norm * r_min

                # Project Δ to satisfy cosine constraint
                if v_j is not None and tau_j is not None:
                    cos_sim = F.cosine_similarity(delta.unsqueeze(0), v_j.unsqueeze(0)).item()
                    if cos_sim < tau_j:
                        v_j_norm = v_j / torch.norm(v_j, p=2)
                        delta = tau_j * torch.norm(delta, p=2) * v_j_norm

                # Update model: θᵢ = θᵗ + Δ
                new_flat = flatten_model(global_model) + delta
                cache_model = unflatten_model(new_flat, global_model)
                cache_model.train()

        # === Step 4: Evaluate ASR before upload ===
        asr = None
        if test_loader is not None:
            cache_model.eval()
            total_correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    # Only use samples not already with target label
                    sample_indices = ~(labels == target_label)
                    inputs = inputs[sample_indices]
                    labels = labels[sample_indices]

                    if len(inputs) == 0:
                        continue

                    # Apply trigger
                    inputs = (1 - mask_) * inputs + mask_ * trigger_
                    inputs = inputs.to(self.params["run_device"])
                    labels_poisoned = torch.full_like(labels, target_label).to(self.params["run_device"])

                    outputs = cache_model(inputs)
                    preds = outputs.argmax(dim=1)

                    total_correct += (preds == labels_poisoned).sum().item()
                    total += inputs.size(0)

            asr = total_correct / total if total > 0 else 0.0
            self.asr_before_upload[client_id] = asr
            logger.info(f"[Iteration {iteration}] Client {client_id} ASR before upload: {asr:.4f}")
        else:
            self.asr_before_upload[client_id] = None  # fallback if test_loader is missing

        return cache_model

