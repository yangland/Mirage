import copy
import torch
import logging
import time

from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from tqdm import tqdm
from participants.clients.BasicClient import BasicClient
from utils.utils import poisoned_batch_injection, test_model_asr_acc
from utils.regoin_utils import flatten_model, compute_geo_loss, project_model_into_region, \
    search_k_percent_to_fix_geometry, apply_delta_to_model, check_cos_constraint, scale_model_update_to_l2_boundary, \
    project_update_inplace_, scale_update_to_l2_boundary_inplace_, _assign_flat_params_
from utils.visualize import visualize, visualize_batch, visualize_tsne
import torch.nn.functional as F

logger = logging.getLogger("logger")


class MirageClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader):
        super(MirageClient, self).__init__(params, train_dataloader, test_dataloader)
        self.init_trigger_mask()
        # {region_id → trigger} Stores trigger/mask for each attacked region, initialized during broadcast_upload
        self.trigger_set_by_region = {}
        self.mask_set_by_region = {}

    def generate_discriminator_dataloader(self, model, train_loader, trigger_, mask_, client_id, region_id):
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

        target_class = self.params["poison_label_swap_by_region"][region_id]

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
            poisoned_sample, _ = poisoned_batch_injection(batch=(samples, labels), 
                                                          trigger=trigger_,
                                                          mask=mask_, 
                                                          is_eval=True,
                                                          client_id=client_id,
                                                          region_id=region_id)
            
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

    def search_trigger(self, model, train_loader, client_id, test_loader=None, region_id=None):
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
            dataloader_discriminator = self.generate_discriminator_dataloader(model=model, 
                                                                              train_loader=local_train_loader, 
                                                                              trigger_=t, 
                                                                              mask_=mask_,
                                                                              client_id=client_id,
                                                                              region_id=region_id)
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

                backdoor_inputs, backdoor_targets = poisoned_batch_injection(batch=(backdoor_inputs, backdoor_targets),
                                                                             trigger=t, 
                                                                             mask=mask_, 
                                                                             is_eval=False,
                                                                             client_id=client_id,
                                                                             region_id=region_id
                                                                             )

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


    def local_train_mirage(self, iteration, model, train_loader, client_id, test_loader=None, region_id=None):
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

        trigger_ = self.search_trigger(model=cache_model, 
                                       train_loader=train_loader, 
                                       client_id=client_id,
                                       region_id=region_id)
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
                inputs, labels = poisoned_batch_injection(batch=batch, 
                                                          trigger=self.trigger_set[client_id],
                                                          mask=self.mask_set[client_id], 
                                                          is_eval=False,
                                                          client_id=client_id,
                                                          region_id=region_id)

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


    def local_train_region_constrained(
        self,
        iteration,
        model,
        train_loader,
        client_id,
        constraint,
        test_loader=None,
        region_id=None
    ):
        """
        Region-constrained local training with geometric loss (for R1–R4).
        """
        device = self.params["run_device"]
        global_model = copy.deepcopy(model)
        cache_model = copy.deepcopy(model)

        optimizer = torch.optim.SGD(
            cache_model.parameters(),
            lr=self.params['poisoned_lr'],
            momentum=self.params['poisoned_momentum'],
            weight_decay=self.params['poisoned_weight_decay']
        )

        # === Step 1: Optimize the Trigger ===
        trigger_ = self.search_trigger(model=cache_model, 
                                train_loader=train_loader, 
                                client_id=client_id,
                                region_id=region_id)
        self.trigger_set[client_id] = trigger_
        mask_ = self.mask_set[client_id]

        ce_loss = nn.CrossEntropyLoss().to(device)

        # === Step 2: Extract region constraint info ===
        avg_benign_model = constraint.get("avg_benign_weight", None)
        l2_radius = constraint.get("l2_radius")
        update_cone_mode = constraint.get("update_cone_mode")
        logger.info(f"client: {client_id}, l2 radius: {l2_radius}, update cone mode: {update_cone_mode}")

        # Precompute Δb = θ̄_b - θ^t
        delta_b = flatten_model(avg_benign_model) - flatten_model(global_model)

        # === Step 3: Training Loop ===
        for epoch in range(self.params["poisoned_retrain_no_times"]):
            for batch_idx, batch in enumerate(train_loader):
                inputs, labels = poisoned_batch_injection(
                    batch=batch,
                    trigger=self.trigger_set[client_id],
                    mask=mask_,
                    is_eval=False,
                    client_id=client_id,
                    region_id=region_id
                )
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = cache_model(inputs)
                base_loss = ce_loss(outputs, labels)

                # print(f"[DEBUG] outputs stats: min={outputs.min().item()}, max={outputs.max().item()}, mean={outputs.mean().item()}")
                # print(f"[DEBUG] labels range: {labels.min().item()} to {labels.max().item()}")
                # print(f"[DEBUG] outputs contains NaN: {torch.isnan(outputs).any().item()}, Inf: {torch.isinf(outputs).any().item()}")

                # Compute Δθ (current update)
                delta_theta = flatten_model(cache_model) - flatten_model(global_model)
                norm_theta = delta_theta.norm().item()

                # === Option B: Only apply geo loss if Δθ is meaningfully non-zero ===
                norm_theta_threshold = 1e-6
                
                if norm_theta > norm_theta_threshold:
                    geo_loss = compute_geo_loss(
                        delta_theta=delta_theta,
                        delta_b=delta_b,
                        update_cone_mode=update_cone_mode,
                        threshold = norm_theta_threshold
                    )
                else:
                    # Skip geo loss if delta_theta is too small
                    geo_loss = torch.tensor(0.0, device=delta_theta.device, requires_grad=True)

                # Total loss
                total_loss = base_loss + self.params.get("lambda_geo", 1.0) * geo_loss

                if torch.isnan(total_loss):
                    print(f"[FATAL] Loss is NaN at iteration {iteration}, client {client_id}")
                    print(f"    base_loss: {base_loss.item()}, geo_loss: {geo_loss.item()}, delta_theta_norm: {norm_theta:.2e}")
                    return model  # Fail-safe

                total_loss.backward()
                optimizer.step()

                # === Project back into L2 region if exceed L2 radius ===
                project_update_inplace_(cache_model, global_model, l2_radius)

        # --- TEST[1] BEFORE final L2 scaling ---
        if test_loader is not None:  # NEW: local eval
            m_pre = test_model_asr_acc(
                model=cache_model,
                test_dataloader=test_loader,
                device=device,
                trigger=trigger_,
                mask=mask_,
                client_id=client_id,
                region_id=region_id,
                poisoned_batch_injection=poisoned_batch_injection
            )
            logger.info(f"[LocalEval][Client {client_id}] pre-scale: "
                        f"clean_acc={m_pre['clean_acc']*100:.2f}%, "
                        f"ASR={m_pre['asr']*100:.2f}%")

        # After poisoned training loop, make sure the L2 norm as required
        scale_update_to_l2_boundary_inplace_(cache_model, global_model, l2_radius)


        # --- TEST[2] AFTER final L2 scaling ---
        if test_loader is not None:  # NEW: local eval
            m_post = test_model_asr_acc(
                model=cache_model,
                test_dataloader=test_loader,
                device=device,
                trigger=trigger_,
                mask=mask_,
                client_id=client_id,
                region_id=region_id,
                poisoned_batch_injection=poisoned_batch_injection
            )
            logger.info(f"[LocalEval][Client {client_id}] post-scale: "
                        f"clean_acc={m_post['clean_acc']*100:.2f}%, "
                        f"ASR={m_post['asr']*100:.2f}%")


        # === Step 5: Validate Geometry ===
        delta_theta = flatten_model(cache_model) - flatten_model(global_model)
        cosine_threshold = constraint.get("cosine_threshold")

        is_valid, crafted_cos_dist  = check_cos_constraint(
            delta_theta=delta_theta,
            delta_b=delta_b,
            update_cone_mode=update_cone_mode,
            cosine_threshold=cosine_threshold
        )

        # === Step 6: Binary Search for Value Replacement if Invalid ===
        if not is_valid:
            logger.warning(
                            f"[Iteration {iteration}] Client {client_id} violates geo constraint. "
                            f"cos_dist = {crafted_cos_dist:.4f}, threshold = {cosine_threshold:.4f}. "
                            "Starting binary search for k% replacement."
                            )
            fixed_delta, final_k = search_k_percent_to_fix_geometry(
                delta_theta=delta_theta,
                delta_b=delta_b,
                update_cone_mode=update_cone_mode,
                cosine_threshold=cosine_threshold
            )

            if final_k is not None:
                logger.info(f"[Fix] Client {client_id} — Constraint fixed with bottom {final_k}% replaced.")
                cache_model = apply_delta_to_model(global_model, fixed_delta)

                # ensure exact L2 radius without changing direction
                scale_update_to_l2_boundary_inplace_(cache_model, global_model, l2_radius)

                # --- TEST[3] AFTER binary replacement + rescale ---
                if test_loader is not None:  # NEW: local eval
                    m_fix = test_model_asr_acc(
                        model=cache_model,
                        test_dataloader=test_loader,
                        device=device,
                        trigger=trigger_,
                        mask=mask_,
                        client_id=client_id,
                        region_id=region_id,
                        poisoned_batch_injection=poisoned_batch_injection
                    )
                    logger.info(f"[LocalEval][Client {client_id}] post-binary-fix: "
                                f"clean_acc={m_fix['clean_acc']*100:.2f}%, "
                                f"ASR={m_fix['asr']*100:.2f}%")

                # optional: re-check
                delta_theta = flatten_model(cache_model) - flatten_model(global_model)
                ok, cos_d = check_cos_constraint(delta_theta, delta_b, update_cone_mode, cosine_threshold)
                if not ok:
                    logger.warning(f"[Fix] Cosine check failed after scaling (cos_dist={cos_d:.4f}).")
            else:
                logger.warning(f"[Fix] Client {client_id} — Could NOT fix constraint with ≤100% flipping.")

        return cache_model


    def local_train(
        self,
        iteration,
        model,
        train_loader,
        client_id,
        test_loader=None,
        region_constraints=None,
        region_id= None,
        attack_variant="region constraint",  # or "Mirage org", "region constraint"
    ):
        """
        Unified local training interface.
        Supports:
        - Benign training
        - Mirage-style poisoning
        - Region-constrained poisoning
        """

        if attack_variant == "Mirage org":
            print(f"[DEBUG] Client {client_id} using 'Mirage org' attack.")
            return self.local_train_mirage(iteration, model, train_loader, client_id, region_id=region_id)

        elif attack_variant == "region constraint":
            print(f"[DEBUG] Client {client_id} using 'region constraint' attack.")
            if region_constraints is None:
                raise ValueError(f"[ERROR] No region constraint found for client {client_id}")
            return self.local_train_region_constrained(
                iteration=iteration,
                model=model,
                train_loader=train_loader,
                client_id=client_id,
                constraint=region_constraints, # constraint for this client
                test_loader=test_loader,
                region_id=region_id
            )
        else:
            raise ValueError(f"[ERROR] Unknown attack_variant: {attack_variant}")




