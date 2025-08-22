import copy
import torch
import logging
import time

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from participants.clients.BasicClient import BasicClient
from utils.utils import poisoned_batch_injection, test_model_asr_acc, _tiny_fp, eval_and_log_local, eval_and_log_local_dual
from utils.regoin_utils import flatten_model, compute_geo_loss, project_model_into_region, \
    search_k_percent_to_fix_geometry, apply_delta_to_model, check_cos_constraint, scale_model_update_to_l2_boundary, \
    project_update_inplace_, scale_update_to_l2_boundary_inplace_, _assign_flat_params_, polish_after_fix
from utils.visualize import visualize, visualize_batch, visualize_tsne

from typing import Iterable, Tuple, Dict, Optional
import torch
import torch.nn as nn
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


    def local_train_mirage(
        self,
        iteration,
        model,
        train_loader,
        client_id,
        server_test_loader=None, 
        mali_test_loader=None,
        region_id=None,
        log_trigger_dbg=False,
    ):
        device = self.params["run_device"]
        global_model = copy.deepcopy(model)
        cache_model  = copy.deepcopy(model)

        # --- FIX: robust loss_fn selection ---
        loss_fn = getattr(self, "criterion", None)
        if isinstance(loss_fn, nn.Module):
            loss_fn = loss_fn.to(device)
        elif callable(loss_fn):
            # keep as-is (functions don't need .to(device))
            pass
        else:
            loss_fn = nn.CrossEntropyLoss().to(device)
        # -------------------------------------

        optimizer = torch.optim.SGD(
            cache_model.parameters(),
            lr=self.params['poisoned_lr'],
            momentum=self.params['poisoned_momentum'],
            weight_decay=self.params['poisoned_weight_decay']
        )

        # Ensure region-level dicts exist (server ASR relies on these)
        if not hasattr(self, "trigger_set_by_region"):
            self.trigger_set_by_region = {}
        if not hasattr(self, "mask_set_by_region"):
            self.mask_set_by_region = {}

        trigger_ = self.search_trigger(
            model=cache_model,
            train_loader=train_loader,
            client_id=client_id,
            region_id=region_id
        )
        self.trigger_set[client_id] = trigger_
        mask_ = self.mask_set[client_id]
        if log_trigger_dbg:
            logger.info(f"[TriggerDbg][Client {client_id} R{region_id}] "
                f"pre-train trig_fp={_tiny_fp(trigger_)}, mask_fp={_tiny_fp(mask_)}")

        self.trigger_set_by_region[region_id] = trigger_
        self.mask_set_by_region[region_id]    = mask_

        cache_model.train()
        for epoch in range(self.params["poisoned_retrain_no_times"]):
            for batch in train_loader:
                inputs, labels = poisoned_batch_injection(
                    batch=batch,
                    trigger=trigger_,
                    mask=mask_,
                    is_eval=False,
                    client_id=client_id,
                    region_id=region_id
                )
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = cache_model(inputs)
                loss = loss_fn(outputs, labels)   # works for module or function
                loss.backward()
                optimizer.step()


        # Optional local eval
        if server_test_loader is not None:
            metrics = test_model_asr_acc(
                model=cache_model,
                test_dataloader=server_test_loader,
                device=device,
                trigger=trigger_,
                mask=mask_,
                client_id=client_id,
                region_id=region_id,
                poisoned_batch_injection=poisoned_batch_injection
            )
            logger.info(f"[MirageEval][Client {client_id}] clean_acc={metrics['clean_acc']*100:.2f}%, "
                        f"ASR={metrics['asr']*100:.2f}%")

        return cache_model



    def local_train_region_constrained(
        self,
        iteration,
        model,
        train_loader,
        client_id,
        constraint,
        server_test_loader=None, 
        mali_test_loader=None,
        region_id=None,
        log_trigger_dbg=False
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
        
        # --- ensure per-client stores are dicts (convert lists on-the-fly) ---
        if isinstance(self.trigger_set, list):
            self.trigger_set = {i: t for i, t in enumerate(self.trigger_set) if t is not None}
        if isinstance(self.mask_set, list):
            self.mask_set = {i: m for i, m in enumerate(self.mask_set) if m is not None}

        # save trigger/mask for this *client* and also for this *region*
        self.trigger_set[client_id] = trigger_
        mask_ = self.mask_set.get(client_id, self.mask_set.get(region_id, None))
        if mask_ is None:
            # if your mask is fixed (e.g., all-ones in a small patch), initialize it here once
            mask_ = self.mask_set[client_id] = self.mask_set.get(client_id, torch.ones_like(trigger_))

        # also index by region so server evaluation can fetch it directly
        self.trigger_set_by_region[region_id] = trigger_
        self.mask_set_by_region[region_id] = mask_

        if log_trigger_dbg:
            logger.info(f"[TriggerDbg][Client {client_id} R{region_id}] "
                        f"pre-train trig_fp={_tiny_fp(trigger_)}, mask_fp={_tiny_fp(mask_)}")

        ce_loss = nn.CrossEntropyLoss().to(device)

        # === Step 2: Extract region constraint info ===
        avg_benign_model = constraint.get("avg_benign_weight", None)
        l2_radius = constraint.get("l2_radius")
        train_radius_scale  = self.params.get("train_radius_scale", 3.0)
        train_radius = l2_radius * train_radius_scale
        project_every = self.params.get("project_every", 3)
        update_cone_mode = constraint.get("update_cone_mode")
        logger.info(f"client: {client_id}, l2 radius: {l2_radius}, update cone mode: {update_cone_mode}")

        # Precompute Δb = θ̄_b - θ^t
        delta_b = flatten_model(avg_benign_model) - flatten_model(global_model)

        # === Step 3: Training Loop ===
        cache_model.train()
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

                # Compute Δθ (current update)
                delta_theta = flatten_model(cache_model) - flatten_model(global_model)
                norm_theta = delta_theta.norm().item()

                # === Direction + Magnitude losses (replaces geo_loss) ===
                norm_theta_threshold = 1e-6
                lambda_dir = self.params.get("lambda_dir", 0.5)   # reuse your knob
                lambda_mag = self.params.get("lambda_mag", 0.1)   # NEW

                # Build target direction u (handles align/oppose via sign)
                if delta_b.norm().item() <= norm_theta_threshold or norm_theta <= norm_theta_threshold:
                    dir_loss = (delta_theta * 0.0).sum()  # graph-connected zero
                    u = None
                else:
                    target = delta_b if update_cone_mode == 1 else -delta_b
                    u = target / (target.norm() + 1e-12)
                    cos_sim = F.cosine_similarity(delta_theta.unsqueeze(0), u.unsqueeze(0), eps=1e-8).clamp(-1.0, 1.0)
                    dir_loss = 1.0 - cos_sim  # smaller is better

                # Magnitude loss toward the region radius (one radius everywhere)
                mag_loss = (delta_theta.norm() - l2_radius).pow(2)

                # Optional schedule: ramp λ_dir during epochs to avoid early tug-of-war
                if self.params.get("use_dir_schedule", True):
                    T = max(1, self.params["poisoned_retrain_no_times"])
                    ramp = float(epoch + 1) / float(T)
                else:
                    ramp = 1.0

                # Total loss
                total_loss = base_loss + (ramp * lambda_dir) * dir_loss + lambda_mag * mag_loss


                if torch.isnan(total_loss):
                    print(f"[FATAL] Loss is NaN at iteration {iteration}, client {client_id}")
                    print(f"    base_loss: {base_loss.item()}, dir_loss: {dir_loss.item()}, mag_loss: {mag_loss.item()}, delta_theta_norm: {norm_theta:.2e}")
                    return model  # Fail-safe

                total_loss.backward()
                
                # === Optional gradient shaping toward the cone ===
                beta = self.params.get("beta_grad_proj", None)  # e.g., 0.5; set None to disable
                if (beta is not None) and (0.0 <= beta < 1.0) and (u is not None):
                    g_list, shapes = [], []
                    for p in cache_model.parameters():
                        if p.grad is None: 
                            continue
                        g_list.append(p.grad.view(-1))
                        shapes.append(p.grad.shape)
                    if g_list:
                        g = torch.cat(g_list)
                        u_hat = u / (u.norm() + 1e-12)  # already unit, safe to re-norm

                        g_par  = torch.dot(g, u_hat) * u_hat
                        g_perp = g - g_par
                        g_new  = g_par + beta * g_perp    # beta=0 -> along u only; beta=1 -> unchanged

                        start = 0
                        for p, shp in zip(cache_model.parameters(), shapes):
                            n = p.numel()
                            p.grad.copy_(g_new[start:start+n].view(shp))
                            start += n

                optimizer.step()

                # === Project back into L2 region if exceed L2 radius ===
                if project_every and ((batch_idx + 1) % project_every == 0):
                    project_update_inplace_(cache_model, global_model, train_radius)

        # --- TEST[1] BEFORE final L2 scaling ---
        m_pre = eval_and_log_local_dual(
            model=cache_model,
            device=device,
            server_test_loader=server_test_loader,
            mali_test_loader=mali_test_loader,
            client_id=client_id,
            region_id=region_id,
            trigger=trigger_,
            mask=mask_,
            poisoned_batch_injection=poisoned_batch_injection,
            logger=logger,
            tag="pre-scale",
            show_trigger_dbg=False,
            views=("malicious",),  # <--- only malicious-side eval
        )

        # After poisoned training loop, make sure the L2 norm as required
        scale_update_to_l2_boundary_inplace_(cache_model, global_model, l2_radius)


        # --- TEST[2] AFTER final L2 scaling ---
        m_post = eval_and_log_local_dual(
            model=cache_model,
            device=device,
            server_test_loader=server_test_loader,
            mali_test_loader=mali_test_loader,
            client_id=client_id,
            region_id=region_id,
            trigger=trigger_,
            mask=mask_,
            poisoned_batch_injection=poisoned_batch_injection,
            logger=logger,
            tag="post-scale",
            show_trigger_dbg=False,   # show the TriggerDbg line once here
            views=("malicious",),  # <--- only malicious-side eval
        )

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
                # 1) Apply replacement and snap to exact radius
                cache_model = apply_delta_to_model(global_model, fixed_delta)
                scale_update_to_l2_boundary_inplace_(cache_model, global_model, l2_radius)

                # 2) Evaluate right after replacement (no polish yet)
                m_repl = eval_and_log_local_dual(
                    model=cache_model,
                    server_test_loader=server_test_loader,
                    mali_test_loader=mali_test_loader,
                    device=device,
                    client_id=client_id,
                    region_id=region_id,
                    trigger=trigger_,
                    mask=mask_,
                    poisoned_batch_injection=poisoned_batch_injection,
                    logger=logger,
                    tag="post-replace",
                    show_trigger_dbg=False,
                    views=("malicious",),  # <--- only malicious-side eval
                )

                # 3) Decide if polish is needed based on ASR drop (malicious view only)
                asr_drop_thresh = float(self.params.get("polish_asr_drop_thresh", 0.10))

                # m_post and m_repl are dual-view dicts → pick the malicious view safely
                m_post_mali = (m_post.get("malicious") or {})
                m_repl_mali = (m_repl.get("malicious") or {})

                post_scale_asr = float(m_post_mali.get("asr") or 0.0)
                post_repl_asr  = float(m_repl_mali.get("asr") or 0.0)
                asr_drop = max(0.0, post_scale_asr - post_repl_asr)

                logger.info(
                    f"[Fix] ASR baseline (post-scale)={post_scale_asr*100:.2f}%, "
                    f"post-replace ASR={post_repl_asr*100:.2f}% "
                    f"(drop={asr_drop*100:.2f}%, thresh={asr_drop_thresh*100:.1f}%)."
                )

                run_polish = asr_drop >= asr_drop_thresh

                if run_polish:
                    # 4) Try polish, but never leave the final model violating cosine constraint.
                    #    Keep a copy to revert if polish exceeds the budget.
                    _pre_polish_sd = {k: v.detach().clone() for k, v in cache_model.state_dict().items()}

                    ok_polish, cos_polish = polish_after_fix(
                        cache_model=cache_model,
                        global_model=global_model,
                        train_loader=train_loader,
                        client_id=client_id,
                        region_id=region_id,
                        trigger=trigger_,
                        mask=mask_,
                        device=device,
                        ce_loss=ce_loss,
                        delta_b=delta_b,
                        update_cone_mode=update_cone_mode,
                        l2_radius=l2_radius,
                        cosine_threshold=cosine_threshold,
                        poisoned_batch_injection=poisoned_batch_injection,
                        project_update_inplace_=project_update_inplace_,
                        scale_update_to_l2_boundary_inplace_=scale_update_to_l2_boundary_inplace_,
                        params=self.params,
                        logger=logger,
                    )

                    if not ok_polish:
                        # Revert to pre-polish weights to respect geometry budget
                        logger.warning(
                            f"[Polish] Reverting polish because cosine exceeded threshold "
                            f"(cos_dist={cos_polish:.4f} > {cosine_threshold:.4f})."
                        )
                        cache_model.load_state_dict(_pre_polish_sd)
                        # (already on-radius from before polishing)

                    # 5) Evaluate after (successful-or-reverted) polish
                    m_fix = eval_and_log_local_dual(
                        model=cache_model,
                        server_test_loader=server_test_loader,
                        mali_test_loader=mali_test_loader,
                        device=device,
                        client_id=client_id,
                        region_id=region_id,
                        trigger=trigger_,
                        mask=mask_,
                        poisoned_batch_injection=poisoned_batch_injection,
                        logger=logger,
                        tag="post-binary-fix",
                        show_trigger_dbg=False,
                        views=("malicious",),  # <--- only malicious-side eval
                    )
                else:
                    logger.info(
                        "[Polish] Skipped: ASR did not drop enough after replacement "
                        f"(drop={asr_drop*100:.2f}% < {asr_drop_thresh*100:.1f}%)."
                    )
                    m_fix = m_repl  # carry forward post-replace metrics as the final

                # 6) Final geometry check (informational)
                delta_theta = flatten_model(cache_model) - flatten_model(global_model)
                ok, cos_d = check_cos_constraint(delta_theta, delta_b, update_cone_mode, cosine_threshold)
                if not ok:
                    logger.warning(
                        f"[Fix] Cosine check failed at end (cos_dist={cos_d:.4f}) "
                        f"vs cos threshold {cosine_threshold:.4f}."
                    )
            else:
                logger.warning(f"[Fix] Client {client_id} — Could NOT fix constraint with ≤100% flipping.")

        return cache_model


    def local_train_pga(
        self,
        iteration,
        model,
        train_loader,
        client_id,
        server_test_loader=None,
        mali_test_loader=None,
        pga_opts: Optional[dict] = None,
    ):
        """
        Preference-Guided Attack (Phase 2), JSON-driven:
        1) Warm-start with malicious objective only to get Δ_task.
        2) Pick direction (aligned/opposed) and radius band (small/large) from JSON.
        3) Enable only needed guidance:
            - Direction guidance iff σ_dir >= σ_norm AND ⟨Δ_task, u*⟩ < 0.
            -> add λ_dir * cos_d(Δ, u*) and minimal replacement until cos_d(Δ, u*) <= 1.
            - Norm guidance iff σ_norm > σ_dir.
            -> per-step projection to ||Δ|| = r*; else one end-of-training scaling to r*.
        Returns the updated local model to send to server.
        """
        # ---------- resolve options ----------
        device = self.params.get("run_device", "cuda")
        opts = pga_opts or {}

        all_region_constraints = opts.get("all_region_constraints") \
            or self.params.get("all_region_constraints") or {}
        pga_prefs = opts.get("pga_prefs") or self.params.get("pga_prefs")
        if pga_prefs is None:
            raise ValueError("[PGA] Missing preference JSON (pga_prefs)")

        # objective: "backdoor" or "untargeted"
        pga_objective = (opts.get("pga_objective") or self.params.get("pga_objective", "backdoor")).lower()

        # base scale (may have been set in main.py) — safe default if absent
        lambda_pref = float(opts.get("lambda_pref", self.params.get("lambda_pref", 1.0)))

        # sensitivities from JSON
        sens = pga_prefs.get("agg_rule_inference", {}).get("sensitivity", {})
        sigma_norm = float(sens.get("norm", 0.5))
        sigma_dir  = float(sens.get("direction", 0.5))

        # region scores from JSON
        per_region = pga_prefs.get("per_region", {})
        if not per_region:
            raise ValueError("[PGA] per_region missing in preference JSON")
        s1 = float(per_region.get("1", {}).get("normalized_median_alpha", 0.0))
        s2 = float(per_region.get("2", {}).get("normalized_median_alpha", 0.0))
        s3 = float(per_region.get("3", {}).get("normalized_median_alpha", 0.0))
        s4 = float(per_region.get("4", {}).get("normalized_median_alpha", 0.0))

        # ---------- grad-safe helpers (define only if missing) ----------
        if "flat_params_with_grad" not in globals():
            def flat_params_with_grad(m: nn.Module) -> torch.Tensor:
                return torch.cat([p.view(-1) for p in m.parameters()])
        if "assign_flat_to_model" not in globals():
            def assign_flat_to_model(m: nn.Module, flat: torch.Tensor) -> None:
                off = 0
                for p in m.parameters():
                    n = p.numel()
                    p.data.copy_(flat[off:off+n].view_as(p))
                    off += n
        if "cos_distance_vec" not in globals():
            def cos_distance_vec(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
                an = a / (a.norm() + eps)
                bn = b / (b.norm() + eps)
                return 1.0 - torch.dot(an, bn)
        if "project_to_l2_vec" not in globals():
            def project_to_l2_vec(delta: torch.Tensor, r: float, eps: float = 1e-12) -> torch.Tensor:
                n = delta.norm() + eps
                return delta if n <= r else delta * (r / n)

        # minimal directional fix: binary search the smallest replacement budget to make dot >= 0
        def minimal_fix_direction(delta_vec: torch.Tensor, u_vec: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                N = delta_vec.numel()
                # order indices by |Δ| ascending
                _, idx = torch.topk(delta_vec.abs(), k=N, largest=False)
                left, right = 0, N
                # quick return if already OK
                if torch.dot(delta_vec, u_vec) >= 0:
                    return delta_vec
                # binary search
                while left < right:
                    mid = (left + right) // 2
                    test = delta_vec.clone()
                    test[idx[:mid]] = u_vec[idx[:mid]]
                    if torch.dot(test, u_vec) >= 0:
                        right = mid
                    else:
                        left = mid + 1
                # apply minimal replacement
                fixed = delta_vec.clone()
                if left > 0:
                    fixed[idx[:left]] = u_vec[idx[:left]]
                return fixed

        # ---------- Step 0: warm-start (malicious objective only) ----------
        ce = torch.nn.CrossEntropyLoss().to(device)
        is_backdoor = (pga_objective == "backdoor")

        global_model = copy.deepcopy(model).to(device)  # anchor: global_model = copy.deepcopy(model)
        global_model.train(True)
        base_vec = flat_params_with_grad(global_model).detach()

        # prepare trigger/mask if backdoor (reuse Phase-1 pipeline if present)
        trigger, mask = None, None
        if is_backdoor:
            if hasattr(self, "trigger_set_by_region"):
                # if you also kept a "top region" cache, you can reuse; else search
                cached_any = next(iter(self.trigger_set_by_region.values()), None) if self.trigger_set_by_region else None
                trigger = cached_any
                mask = getattr(self, "mask_set_by_region", {}).get(next(iter(self.mask_set_by_region), None), None) if hasattr(self, "mask_set_by_region") else None
            if trigger is None and hasattr(self, "search_trigger"):
                trigger = self.search_trigger(model=global_model, train_loader=train_loader,
                                            client_id=client_id, region_id=None)
                if not hasattr(self, "trigger_set_by_region"):
                    self.trigger_set_by_region = {}
                    self.mask_set_by_region = {}
                self.trigger_set_by_region[-1] = trigger
                if mask is None:
                    mask = torch.ones_like(trigger)
                    self.mask_set_by_region[-1] = mask

        # small number of warm steps
        warm_steps = int(self.params.get("pga_warm_steps", 1))
        opt_warm = torch.optim.SGD(
            global_model.parameters(),
            lr=self.params.get('poisoned_lr', 0.05),
            momentum=self.params.get('poisoned_momentum', 0.0),
            weight_decay=self.params.get('poisoned_weight_decay', 5e-4),
        )
        if warm_steps > 0:
            it_warm = 0
            for _ in range(warm_steps):
                for batch in train_loader:
                    if is_backdoor and hasattr(self, "poisoned_batch_injection"):
                        inputs, labels = self.poisoned_batch_injection(
                            batch=batch, trigger=trigger, mask=mask,
                            is_eval=False, client_id=client_id, region_id=None
                        )
                    else:
                        inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    opt_warm.zero_grad()
                    logits = global_model(inputs)
                    loss_obj = ce(logits, labels) if is_backdoor else -ce(logits, labels)
                    loss_obj.backward()
                    opt_warm.step()
                    it_warm += 1
                    # one or two batches are enough to probe direction
                    if it_warm >= max(1, self.params.get("pga_warm_batches", 1)):
                        break
                if it_warm >= max(1, self.params.get("pga_warm_batches", 1)):
                    break

        theta_after_warm = flat_params_with_grad(global_model).detach()
        delta_task = theta_after_warm - base_vec  # Δ_task

        # ---------- benign reference update Δ_b ----------
        # Prefer server-provided benign mean in any region constraint; else quick benign step
        delta_b = None
        for rid_try in (1, 2, 3, 4):
            rc = all_region_constraints.get(rid_try, {})
            if "avg_benign_weight" in rc and rc["avg_benign_weight"] is not None:
                delta_b = flat_params_with_grad(rc["avg_benign_weight"]).to(device) - base_vec
                break
        if delta_b is None:
            benign_m = copy.deepcopy(model).to(device)
            benign_m.train(True)
            opt_b = torch.optim.SGD(
                benign_m.parameters(),
                lr=self.params.get('benign_lr', 0.1),
                momentum=self.params.get('benign_momentum', 0.9),
                weight_decay=self.params.get('benign_weight_decay', 5e-4),
            )
            # one quick batch
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt_b.zero_grad()
                out_b = benign_m(xb)
                F.cross_entropy(out_b, yb).backward()
                opt_b.step()
                break
            delta_b = flat_params_with_grad(benign_m).to(device) - base_vec

        # ---------- Step 1: pick direction sign and radius band from JSON ----------
        s_aligned = max(s1, s2)
        s_opposed = max(s3, s4)
        align_sign = +1 if s_aligned >= s_opposed else -1
        u_ref = align_sign * delta_b

        s_small = max(s1, s3)
        s_large = max(s2, s4)

        # collect band radii from constraints; fallback to YAML scales * ||Δ_b||
        def band_radius_from_constraints(ids):
            vals = []
            for rid_try in ids:
                rc = all_region_constraints.get(rid_try, {})
                if "l2_radius" in rc:
                    vals.append(float(rc["l2_radius"]))
            return (sum(vals) / len(vals)) if len(vals) > 0 else None

        r_small = band_radius_from_constraints([1, 3])
        r_large = band_radius_from_constraints([2, 4])

        if r_small is None or r_large is None:
            # fallback using YAML scales and benign norm
            scales_S = self.params.get("l2_scale_min", [0.5, 1.2])
            scales_L = self.params.get("l2_scale_max", [3.0, 5.0])
            bnorm = float(delta_b.norm().item())
            r_small = r_small or (0.5 * (scales_S[0] + scales_S[1]) * bnorm)
            r_large = r_large or (0.5 * (scales_L[0] + scales_L[1]) * bnorm)

        use_large_band = (s_large > s_small)
        r_star = float(r_large if use_large_band else r_small)

        # ---------- Step 2: enable only what is needed ----------
        # direction enabled iff σ_dir >= σ_norm AND Δ_task in wrong half-space
        dir_enabled = (sigma_dir >= sigma_norm) and (torch.dot(delta_task, u_ref) < 0)
        # norm enabled iff σ_norm > σ_dir
        norm_enabled = (sigma_norm > sigma_dir)

        lam_dir = (lambda_pref * sigma_dir) if dir_enabled else 0.0
        lam_mag = (lambda_pref * sigma_norm) if norm_enabled else 0.0

        logger.info(f"[PGA] dir_enabled={dir_enabled}, norm_enabled={norm_enabled}, "
                    f"align_sign={align_sign}, band={'L' if use_large_band else 'S'}, r*={r_star:.4f}")

        # ---------- Step 3: final optimization with needed terms only ----------
        opt = torch.optim.SGD(
            global_model.parameters(),
            lr=self.params.get('poisoned_lr', 0.05),
            momentum=self.params.get('poisoned_momentum', 0.0),
            weight_decay=self.params.get('poisoned_weight_decay', 5e-4),
        )
        project_every = int(self.params.get("project_every", 2))
        steps = int(self.params.get("poisoned_retrain_no_times", 5))

        for epoch in range(steps):
            for bidx, batch in enumerate(train_loader):

                # build batch
                if is_backdoor and hasattr(self, "poisoned_batch_injection"):
                    inputs, labels = self.poisoned_batch_injection(
                        batch=batch, trigger=trigger, mask=mask,
                        is_eval=False, client_id=client_id, region_id=None
                    )
                else:
                    inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                # forward & task loss
                opt.zero_grad()
                logits = global_model(inputs)
                loss_obj = ce(logits, labels) if is_backdoor else -ce(logits, labels)

                # current Δ
                theta_vec = flat_params_with_grad(global_model)
                delta_vec = theta_vec - base_vec

                # preference penalties (only the enabled ones)
                loss = loss_obj
                if lam_dir > 0.0:
                    loss = loss + lam_dir * cos_distance_vec(delta_vec, u_ref)
                if lam_mag > 0.0:
                    loss = loss + lam_mag * (delta_vec.norm() - r_star) ** 2

                loss.backward()
                opt.step()

                # norm guidance: project each step
                if norm_enabled:
                    theta_vec = flat_params_with_grad(global_model)
                    delta_vec = theta_vec - base_vec
                    delta_vec = project_to_l2_vec(delta_vec, r_star)
                    assign_flat_to_model(global_model, base_vec + delta_vec)

                # direction guidance: minimal repair only if still wrong half-space
                if lam_dir > 0.0:
                    theta_vec = flat_params_with_grad(global_model)
                    delta_vec = theta_vec - base_vec
                    if torch.dot(delta_vec, u_ref) < 0:
                        delta_vec = minimal_fix_direction(delta_vec, u_ref)
                        if norm_enabled:
                            delta_vec = project_to_l2_vec(delta_vec, r_star)
                        assign_flat_to_model(global_model, base_vec + delta_vec)

                if project_every and ((bidx + 1) % project_every == 0) and norm_enabled:
                    theta_vec = flat_params_with_grad(global_model)
                    delta_vec = theta_vec - base_vec
                    delta_vec = project_to_l2_vec(delta_vec, r_star)
                    assign_flat_to_model(global_model, base_vec + delta_vec)

        # end-of-training scaling if norm not enabled
        if not norm_enabled:
            theta_vec = flat_params_with_grad(global_model)
            delta_vec = theta_vec - base_vec
            if delta_vec.norm().item() > 0:
                delta_vec = project_to_l2_vec(delta_vec, r_star)  # single scaling
                assign_flat_to_model(global_model, base_vec + delta_vec)

        # optional eval logging (unchanged)
        try:
            if server_test_loader is not None:
                if is_backdoor and hasattr(self, "test_model_asr_acc"):
                    metrics = self.test_model_asr_acc(
                        model=global_model, test_dataloader=server_test_loader, device=device,
                        trigger=trigger, mask=mask, client_id=client_id, region_id=None,
                        poisoned_batch_injection=getattr(self, "poisoned_batch_injection", None)
                    )
                    logger.info(f"[PGA][Client {client_id}] clean_acc={metrics.get('clean_acc',0)*100:.2f}%, ASR={metrics.get('asr',0)*100:.2f}%")
                elif hasattr(self, "eval_acc"):
                    acc = self.eval_acc(global_model, server_test_loader, device=device)
                    logger.info(f"[PGA][Client {client_id}] clean_acc={acc*100:.2f}% (untargeted)")
        except Exception as e:
            logger.warning(f"[PGA] Eval skipped: {e}")

        return global_model


    def local_train(
        self,
        iteration,
        model,
        train_loader,
        client_id,
        server_test_loader=None,
        mali_test_loader=None,
        region_constraints=None,
        region_id=None,
        attack_variant="region constraint",  # or "Mirage org", "region constraint"
        pga_opts: Optional[dict] = None,
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
            return self.local_train_mirage(
                iteration=iteration,
                model=model,
                train_loader=train_loader,
                client_id=client_id,
                server_test_loader=server_test_loader,
                mali_test_loader=mali_test_loader,
                region_id=region_id,
                log_trigger_dbg=self.params.get("log_trigger_dbg", False)
            )

        elif attack_variant == "region constraint":
            print(f"[DEBUG] Client {client_id} using 'region constraint' attack.")
            if region_constraints is None:
                raise ValueError(f"[ERROR] No region constraint found for client {client_id}")
            return self.local_train_region_constrained(
                iteration=iteration,
                model=model, # local model to train
                train_loader=train_loader,
                client_id=client_id,
                constraint=region_constraints, # constraint for this client
                server_test_loader=server_test_loader,
                mali_test_loader=mali_test_loader,
                region_id=region_id,
                log_trigger_dbg=self.params.get("log_trigger_dbg", False)
            )
        elif attack_variant == "pga":
            logger.info(f"[PGA] Client {client_id} entering Preference-Guided Attack")
            return self.local_train_pga(
                iteration=iteration,
                model=model,
                train_loader=train_loader,
                client_id=client_id,
                server_test_loader=server_test_loader,
                mali_test_loader=mali_test_loader,
                pga_opts=pga_opts,
            )
        else:
            raise ValueError(f"[ERROR] Unknown attack_variant: {attack_variant}")




# --- PGA grad-safe helpers (ADD only if you don't already have equivalents) ---
def flat_params_with_grad(model: nn.Module) -> torch.Tensor:
    """Flatten parameters without .data so geometry penalties remain differentiable."""
    return torch.cat([p.view(-1) for p in model.parameters()])

def assign_flat_to_model(model: nn.Module, flat_tensor: torch.Tensor) -> None:
    """In-place copy of a flat vector back into model.params (keeps optimizer state)."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_tensor[offset:offset+n].view_as(p))
        offset += n

def cos_distance_vec(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine distance between two parameter vectors, differentiable."""
    an = a / (a.norm() + eps)
    bn = b / (b.norm() + eps)
    return 1.0 - torch.dot(an, bn)  # 0 aligned, 2 opposite

def project_to_l2_vec(delta: torch.Tensor, r: float, eps: float = 1e-12) -> torch.Tensor:
    n = delta.norm() + eps
    return delta if n <= r else delta * (r / n)

def bottom_k_align_vec(delta: torch.Tensor, ref: torch.Tensor, k_percent: float, align_sign: int) -> torch.Tensor:
    k = max(1, int(delta.numel() * (k_percent / 100.0)))
    _, idx = torch.topk(delta.abs(), k, largest=False)
    out = delta.clone()
    out[idx] = align_sign * ref[idx]
    return out
