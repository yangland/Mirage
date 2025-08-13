import copy
import random
import logging
import time
import itertools
from tqdm import tqdm
import torch

from participants.servers.BasicServer import BasicServer
from utils.utils import model_dist_norm_var, update_weight_accumulator, model_weight_diff, \
    update_weight_accumulator_direct, _tiny_fp
from utils.regoin_utils import compute_benign_statistics, build_region_constraints
import random

logger = logging.getLogger("logger")


class No_defense_Server(BasicServer):
    def __init__(self, params, dataloader, full_train_dataset=None):
        super(No_defense_Server, self).__init__(params, dataloader, full_train_dataset)


    def _clone_trainable_model(self):
        """Deep-copy global model and make it trainable."""
        m = copy.deepcopy(self.global_model)
        for p in m.parameters():
            p.requires_grad = True
        m.train()
        return m

    def _shadow_craft_trigger_for_region(
        self,
        iteration: int,
        malicious_client,
        real_client_id: int,
        region_id: int,
        region_constraints: dict,
        train_loader,
        test_loader,
        logger,
    ) -> bool:
        """
        Run ONE malicious local_train on a temp model to ensure trigger/mask exist
        for this region+client. The update is discarded; we only want trigger/mask.
        """
        tmp = self._clone_trainable_model()

        # Malicious local_train with region constraints to populate trigger/mask stores
        _ = malicious_client.local_train(
            iteration=iteration,
            model=tmp,
            train_loader=train_loader,
            client_id=real_client_id,
            test_loader=test_loader,
            region_constraints=region_constraints,
            region_id=region_id,
        )

        # Copy per-client trigger/mask → per-region stores (if available)
        trig = getattr(malicious_client, "trigger_set", {}).get(real_client_id, None)
        msk  = getattr(malicious_client, "mask_set", {}).get(real_client_id, None)
        if trig is not None and msk is not None:
            malicious_client.trigger_set_by_region[region_id] = trig
            malicious_client.mask_set_by_region[region_id]    = msk
            logger.info(f"[ShadowCraft] R{region_id} ready (trig_fp={_tiny_fp(trig):.3f}, mask_fp={_tiny_fp(msk):.3f})")
            return True

        logger.warning(f"[ShadowCraft] R{region_id}: trigger/mask not produced.")
        return False

    def _warmup_bootstrap_triggers_for_all_regions(
        self,
        iteration: int,
        malicious_client,
        canonical_client_for_region: dict,
        region_constraints_dict: dict,
        logger
    ):
        """
        During warm-up rounds: ensure trigger/mask exist for ALL regions by running
        one malicious local_train per region on its canonical client.
        """
        for rid, real_id in canonical_client_for_region.items():
            if (rid in malicious_client.trigger_set_by_region
                and rid in malicious_client.mask_set_by_region):
                continue  # already have it

            logger.info(f"[Warmup] Bootstrapping trigger/mask for Region {rid} (canonical client {real_id})")
            train_loader = self.train_dataloader[real_id]
            self._shadow_craft_trigger_for_region(
                iteration=iteration,
                malicious_client=malicious_client,
                real_client_id=real_id,
                region_id=rid,
                region_constraints=region_constraints_dict.get(rid, {}),
                train_loader=train_loader,
                test_loader=self.test_dataloader,
                logger=logger,
            )

    def _safe_get_from_store(self, store, key):
        """Read trigger/mask whether store is a dict or a list; else None."""
        if isinstance(store, dict):
            return store.get(key, None)
        if isinstance(store, list):
            if isinstance(key, int) and 0 <= key < len(store):
                return store[key]
            return None
        return None

    def _malicious_client_step(
        self,
        *,
        iteration: int,
        client_id: int,                 # virtual id (e.g., 20100)
        real_id: int,                   # real id (e.g., 1)
        local_model: torch.nn.Module,
        train_loader,
        region_id: int,
        attack_enabled: bool,
        malicious_client,
        benign_client,
        region_constraints_dict: dict,
        weight_accumulator: dict,
        malicious_update_cache: dict,
        already_trained_malicious_clients: set,
        test_loader,
        logger,
    ):
        """
        One step for a malicious client (attack or benign-as-needed).
        Returns: single_wa (dict), update_norm (float), weight_accumulator (dict)
        """
        # Reuse if this real client already crafted an update this round
        if real_id in already_trained_malicious_clients:
            logger.info(f"[Reuse] Iter {iteration}: malicious client {real_id}")
            cached = malicious_update_cache[real_id]
            return cached['update'], cached['norm'], weight_accumulator

        # If attacking is disabled (policy) => train like benign, but ensure region triggers exist.
        if (not attack_enabled) and self.params.get("allow_malicious_as_benign", True):
            # Make sure trigger/mask exist so ASR can be evaluated later
            if region_id not in malicious_client.trigger_set_by_region:
                self._shadow_craft_trigger_for_region(
                    iteration=iteration,
                    malicious_client=malicious_client,
                    real_client_id=real_id,
                    region_id=region_id,
                    region_constraints=region_constraints_dict.get(region_id, {}),
                    train_loader=train_loader,
                    test_loader=test_loader,
                    logger=logger,
                )

            # Train with benign routine
            updated_model = benign_client.local_train(
                iteration=iteration,
                model=local_model,
                train_loader=train_loader,
                client_id=real_id
            )

            update_norm = model_dist_norm_var(updated_model, self.create_global_model_copy()).item()
            weight_accumulator, single_wa = update_weight_accumulator(
                model=updated_model,
                global_model=copy.deepcopy(self.global_model),
                weight_accumulator=weight_accumulator
            )
        else:
            # Normal malicious training (region-constrained crafting)
            updated_model = malicious_client.local_train(
                iteration=iteration,
                model=local_model,
                train_loader=train_loader,
                client_id=real_id,
                test_loader=test_loader,
                region_constraints=region_constraints_dict.get(region_id, {}),
                region_id=region_id
            )
            single_wa = model_weight_diff(
                after=updated_model.state_dict(),
                before=self.global_model.state_dict()
            )
            update_norm = model_dist_norm_var(updated_model, self.create_global_model_copy()).item()
            weight_accumulator = update_weight_accumulator_direct(single_wa, weight_accumulator)

        # cache and mark
        malicious_update_cache[real_id] = {'update': single_wa, 'norm': round(update_norm, 6)}
        already_trained_malicious_clients.add(real_id)

        return single_wa, update_norm, weight_accumulator

    def _benign_client_step(
        self,
        *,
        iteration: int,
        client_id: int,
        local_model: torch.nn.Module,
        train_loader,
        benign_client,
        weight_accumulator: dict,
    ):
        """
        One step for a benign client.
        Returns: single_wa (dict), update_norm (float), weight_accumulator (dict)
        """
        updated_model = benign_client.local_train(
            iteration=iteration,
            model=local_model,
            train_loader=train_loader,
            client_id=client_id
        )
        update_norm = model_dist_norm_var(updated_model, self.create_global_model_copy()).item()
        weight_accumulator, single_wa = update_weight_accumulator(
            model=updated_model,
            global_model=copy.deepcopy(self.global_model),
            weight_accumulator=weight_accumulator
        )
        if not isinstance(single_wa, dict):
            raise RuntimeError(f"[FATAL] Client {client_id} returned non-dict update: {type(single_wa)}")

        return single_wa, update_norm, weight_accumulator

    def _aggregate_triggers_for_regions(
        self,
        *,
        selected_clients_list,
        malicious_clients_list,
        client_region_mapping,
        canonical_client_for_region,
        malicious_client,
        logger,
    ):
        """
        Step X: Aggregate trigger/mask per region (defensive & list/dict aware).
        """
        for client_id in selected_clients_list:
            if client_id not in malicious_clients_list:
                continue

            region_id = client_region_mapping.get(client_id)
            if region_id is None:
                continue

            # Skip if already set for this region this round
            if (region_id in malicious_client.trigger_set_by_region and
                region_id in malicious_client.mask_set_by_region):
                continue

            # Prefer the canonical client's trigger/mask for the region
            canonical_client = canonical_client_for_region.get(region_id, None)
            trig_src = self._safe_get_from_store(getattr(malicious_client, "trigger_set", {}), canonical_client)
            mask_src = self._safe_get_from_store(getattr(malicious_client, "mask_set", {}), canonical_client)

            # Fallback: any malicious client from this round
            if trig_src is None or mask_src is None:
                for m_id in malicious_clients_list:
                    trig_src = self._safe_get_from_store(malicious_client.trigger_set, m_id)
                    mask_src = self._safe_get_from_store(malicious_client.mask_set, m_id)
                    if trig_src is not None and mask_src is not None:
                        logger.warning(f"[TriggerSet] Using fallback trigger/mask from client {m_id} for region {region_id}.")
                        break

            if trig_src is None or mask_src is None:
                logger.warning(f"[TriggerSet] No trigger/mask available for region {region_id} this round.")
                continue

            malicious_client.trigger_set_by_region[region_id] = trig_src
            malicious_client.mask_set_by_region[region_id]    = mask_src
            logger.info(f"[TriggerSet] R{region_id} trig_fp={_tiny_fp(trig_src):.3f}, mask_fp={_tiny_fp(mask_src):.3f}")



    def broadcast_upload(
        self,
        iteration,
        benign_client,
        malicious_client,
        selected_clients_list,
        malicious_clients_list,
        client_region_mapping,
        canonical_client_for_region,
        malicious_client_mapping,
        attack_enable_by_client=None,
        **kwargs
    ):
        logger.info(f"Training on global iteration {iteration} ")
        attack_enable_by_client = attack_enable_by_client or {}

        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = {}
        update_norm_by_client = {}
        global_model = copy.deepcopy(self.global_model)
        aggregated_model_id = [1] * self.params["no_of_participants_per_iteration"]

        # === Step 1: simulate benign models from malicious pool for region stats ===
        benign_sample_num = self.params.get("benign_sample_for_region")
        benign_like_malicious_ids = random.sample(
            self.total_malicious_clients,
            min(benign_sample_num, len(self.total_malicious_clients))
        )
        benign_models_from_malicious = []
        for client_id in benign_like_malicious_ids:
            benign_like_model = self._clone_trainable_model()
            trained_model = benign_client.local_train(
                iteration, benign_like_model, self.train_dataloader[client_id], client_id
            )
            benign_models_from_malicious.append(trained_model)

        # === Step 2: region constraints ===
        if len(benign_models_from_malicious) > 1:
            region_stats = compute_benign_statistics(benign_models_from_malicious, global_model)
            logger.info("[DEBUG] --- Region Statistics Computation ---")
            logger.info(f"[DEBUG] # of simulated benign models: {len(benign_models_from_malicious)}")
            logger.info(f"[DEBUG] Mean pairwise L2 distance: {region_stats['avg_L2_dist']:.4f}")
            logger.info(f"[DEBUG] Mean L2 norm of updates: {region_stats['avg_L2_norm']:.4f}")
            logger.info(f"[DEBUG] Mean cos dist between updates: {region_stats['avg_update_cos_d']:.8f}")
            logger.info(f"[DEBUG] Mean cos dist between updates and benign: {region_stats['avg_update_cos_d_to_benign']:.8f}")
            logger.info(f"[DEBUG] Mean cos dist between weights: {region_stats['avg_weight_cos_d']:.8f}")

            region_constraints_dict = build_region_constraints(
                stats=region_stats,
                l2_scale_min=self.params.get("l2_scale_min"),
                l2_scale_max=self.params.get("l2_scale_max"),
                cos_scale_min=self.params.get("cos_scale_min"),
                logger=logger
            )
        else:
            logger.warning("Not enough benign-like models to compute region statistics.")
            region_constraints_dict = {i: {} for i in range(8)}

        # === Step 3: logging for regions selected/attacked this round ===
        logger.info(f"[Round {iteration}] Using externally provided region assignments: {client_region_mapping}")
        logger.info(f"[Round {iteration}] selected_clients_list: {selected_clients_list}")
        attacked_regions_this_round = sorted({rid for cid, rid in client_region_mapping.items() if cid in selected_clients_list})
        if attacked_regions_this_round:
            for rid in attacked_regions_this_round:
                c = region_constraints_dict.get(rid, {})
                if c:
                    mode = "align" if c["update_cone_mode"] == 1 else "oppose"
                    logger.info(
                        f"[Round {iteration}] R{rid}: l2_radius={c['l2_radius']:.6f}, "
                        f"mode={mode}, cosine_threshold={c['cosine_threshold']:.6f}, "
                        f"l2_scale={c.get('l2_scale', float('nan')):.6f}, "
                        f"cos_scale={c.get('cos_scale', None)}"
                    )
        else:
            logger.info(f"[Round {iteration}] No malicious regions attacked this round.")

        # === Warm-up: ensure triggers for ALL regions if requested ===
        if iteration < self.params.get("warmup_no_attack_iters", 0):
            self._warmup_bootstrap_triggers_for_all_regions(
                iteration=iteration,
                malicious_client=malicious_client,
                canonical_client_for_region=canonical_client_for_region,
                region_constraints_dict=region_constraints_dict,
                logger=logger
            )

        # === caches for malicious clients
        malicious_update_cache = {}
        already_trained_malicious_clients = set()

        # === Step 4: per-client training (malicious/benign)
        for client_id in tqdm(selected_clients_list):
            real_id = malicious_client_mapping.get(client_id, client_id)
            train_loader = self.train_dataloader[real_id]
            local_model = self._clone_trainable_model()

            if client_id in malicious_clients_list:
                region_id = client_region_mapping.get(client_id)
                attack_enabled = bool(attack_enable_by_client.get(client_id, True))
                single_wa, update_norm, weight_accumulator = self._malicious_client_step(
                    iteration=iteration,
                    client_id=client_id,
                    real_id=real_id,
                    local_model=local_model,
                    train_loader=train_loader,
                    region_id=region_id,
                    attack_enabled=attack_enabled,
                    malicious_client=malicious_client,
                    benign_client=benign_client,
                    region_constraints_dict=region_constraints_dict,
                    weight_accumulator=weight_accumulator,
                    malicious_update_cache=malicious_update_cache,
                    already_trained_malicious_clients=already_trained_malicious_clients,
                    test_loader=self.test_dataloader,
                    logger=logger,
                )
            else:
                single_wa, update_norm, weight_accumulator = self._benign_client_step(
                    iteration=iteration,
                    client_id=client_id,
                    local_model=local_model,
                    train_loader=train_loader,
                    benign_client=benign_client,
                    weight_accumulator=weight_accumulator,
                )

            update_norm_by_client[client_id] = round(update_norm, 6)
            weight_accumulator_by_client[client_id] = single_wa
            del local_model

        # Log per-client norms
        for cid in selected_clients_list:
            logger.info(f"Client {cid} update norm: {update_norm_by_client[cid]}")

        # === Step X: fill trigger/mask per region for ASR testing
        self._aggregate_triggers_for_regions(
            selected_clients_list=selected_clients_list,
            malicious_clients_list=malicious_clients_list,
            client_region_mapping=client_region_mapping,
            canonical_client_for_region=canonical_client_for_region,
            malicious_client=malicious_client,
            logger=logger,
        )

        return (
            weight_accumulator,
            weight_accumulator_by_client,
            aggregated_model_id,
            region_constraints_dict
        )


"""
    def broadcast_upload_old(
        self,
        iteration,
        benign_client,
        malicious_client,
        selected_clients_list,
        malicious_clients_list,
        client_region_mapping,
        canonical_client_for_region,
        malicious_client_mapping,
        attack_enable_by_client=None,
        **kwargs
    ):
        logger.info(f"Training on global iteration {iteration} ")
        attack_enable_by_client = attack_enable_by_client or {}
        current_no_of_adversaries = sum([1 for client_id in selected_clients_list if client_id in malicious_clients_list])

        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = {}
        update_norm_by_client = {}
        global_model = copy.deepcopy(self.global_model)
        aggregated_model_id = [1] * self.params["no_of_participants_per_iteration"]

        # === Step 1: Sample t malicious clients for benign-style training ===
        benign_sample_num = self.params.get("benign_sample_for_region")
        benign_like_malicious_ids = random.sample(
            self.total_malicious_clients,
            min(benign_sample_num, len(self.total_malicious_clients))
        )

        benign_models_from_malicious = []

        for client_id in benign_like_malicious_ids:
            benign_like_model = copy.deepcopy(self.global_model)
            for name, param in benign_like_model.named_parameters():
                param.requires_grad = True
            benign_like_model.train()

            trained_model = benign_client.local_train(
                iteration, 
                benign_like_model, 
                self.train_dataloader[client_id], 
                client_id
            )
            benign_models_from_malicious.append(trained_model)

        # === Step 2: Region Statistics ===
        region_constraints_dict = {}
        if len(benign_models_from_malicious) > 1:
            region_stats = compute_benign_statistics(benign_models_from_malicious, global_model)
            logger.info("[DEBUG] --- Region Statistics Computation ---")
            logger.info(f"[DEBUG] # of simulated benign models: {len(benign_models_from_malicious)}")
            logger.info(f"[DEBUG] Mean pairwise L2 distance: {region_stats['avg_L2_dist']:.4f}")
            logger.info(f"[DEBUG] Mean L2 norm of updates: {region_stats['avg_L2_norm']:.4f}")
            logger.info(f"[DEBUG] Mean cos dist between updates: {region_stats['avg_update_cos_d']:.8f}")
            logger.info(f"[DEBUG] Mean cos dist between updates and benign: {region_stats['avg_update_cos_d_to_benign']:.8f}")
            logger.info(f"[DEBUG] Mean cos dist between weights: {region_stats['avg_weight_cos_d']:.8f}")

            region_constraints_dict = build_region_constraints(
                stats=region_stats,
                l2_scale_min=self.params.get("l2_scale_min"),
                l2_scale_max=self.params.get("l2_scale_max"),
                cos_scale_min=self.params.get("cos_scale_min"),
                logger=logger
            )
        else:
            logger.warning("Not enough benign-like models to compute region statistics.")
            region_constraints_dict = {i: {} for i in range(8)}

        # === Step 3: Logging region assignments ===
        logger.info(f"[Round {iteration}] Using externally provided region assignments: {client_region_mapping}")
        logger.info(f"[Round {iteration}] selected_clients_list: {selected_clients_list}")

        # Log constraints ONLY for regions actually attacked this round
        attacked_regions_this_round = sorted({
            rid for cid, rid in client_region_mapping.items()
            if cid in selected_clients_list
        })

        if not attacked_regions_this_round:
            logger.info(f"[Round {iteration}] No malicious regions attacked this round.")
        else:
            for rid in attacked_regions_this_round:
                c = region_constraints_dict.get(rid, {})
                if not c:
                    continue
                mode = "align" if c["update_cone_mode"] == 1 else "oppose"
                logger.info(
                    f"[Round {iteration}] R{rid}: l2_radius={c['l2_radius']:.6f}, "
                    f"mode={mode}, cosine_threshold={c['cosine_threshold']:.6f}, "
                    f"l2_scale={c.get('l2_scale', float('nan')):.6f}, "
                    f"cos_scale={c.get('cos_scale', None)}"
                )


        # === Cache for already trained malicious clients ===
        malicious_update_cache = {}
        already_trained_malicious_clients = set()

        # === Step 4: Train clients ===
        for client_id in tqdm(selected_clients_list):
            real_id = malicious_client_mapping.get(client_id, client_id)
            client_train_data = self.train_dataloader[real_id]

            # Always create local_model
            local_model = copy.deepcopy(self.global_model)
            for name, param in local_model.named_parameters():
                param.requires_grad = True
            local_model.train()

            # === Malicious client logic ===
            if client_id in malicious_clients_list:
                if real_id in already_trained_malicious_clients:
                    logger.info(f"[Round {iteration}] Reusing cached update for malicious client {real_id}")
                    single_wa = malicious_update_cache[real_id]['update']
                    update_norm = malicious_update_cache[real_id]['norm']
                else:
                    region_id = client_region_mapping.get(client_id)

                    updated_model = malicious_client.local_train(
                        iteration=iteration,
                        model=local_model,
                        train_loader=client_train_data,
                        client_id=real_id,
                        test_loader=self.test_dataloader,
                        region_constraints=region_constraints_dict.get(region_id, {}),
                        region_id=region_id
                    )

                    single_wa = model_weight_diff(
                        after=updated_model.state_dict(),
                        before=self.global_model.state_dict()
                    )

                    update_norm = model_dist_norm_var(updated_model, self.create_global_model_copy()).item()
                    malicious_update_cache[real_id] = {
                        'update': single_wa,
                        'norm': round(update_norm, 6)
                    }
                    already_trained_malicious_clients.add(real_id)

                weight_accumulator = update_weight_accumulator_direct(single_wa, weight_accumulator)
                update_norm_by_client[client_id] = round(update_norm, 6)
                weight_accumulator_by_client[client_id] = single_wa

            # === Benign client logic ===
            else:
                updated_model = benign_client.local_train(
                    iteration=iteration,
                    model=local_model,
                    train_loader=client_train_data,
                    client_id=client_id
                )

                update_norm = model_dist_norm_var(updated_model, self.create_global_model_copy()).item()
                update_norm_by_client[client_id] = round(update_norm, 6)

                weight_accumulator, single_wa = update_weight_accumulator(
                    model=updated_model,
                    global_model=copy.deepcopy(self.global_model),
                    weight_accumulator=weight_accumulator
                )

                if not isinstance(single_wa, dict):
                    print(f"[FATAL] Client {client_id} returned non-dict update: {type(single_wa)}")
                    raise RuntimeError("Abort: client update is invalid")

                weight_accumulator_by_client[client_id] = single_wa

            # === Debug info ===
            # if client_id == 0:
            #     print("[DEBUG] Checking update flow for client 0")
            #     print("Type of updated_model:", type(updated_model))
            #     print("Type of global_model:", type(self.global_model))
            #     print("State dict keys:", list(updated_model.state_dict().keys())[:5])
            #     print("Update norm:", update_norm)

            # if isinstance(single_wa, dict):
            #     print(f"[DEBUG] Client {client_id} update keys: {list(single_wa.keys())[:5]}")
            # else:
            #     print(f"[ERROR] Client {client_id}'s update is not a dict → it is {type(single_wa)}: {single_wa}")

            del local_model

        # Log norms
        for client_ind, client_id in enumerate(selected_clients_list):
            logger.info(f"Client {client_id} update norm: {update_norm_by_client[client_id]}")


        # === Step X: Aggregate trigger/mask per region (defensive & list/dict aware) ===
        def _get_from_store(store, key):
            if isinstance(store, dict):
                return store.get(key, None)
            if isinstance(store, list):
                if isinstance(key, int) and 0 <= key < len(store):
                    return store[key]
                return None
            return None  # unknown type

        for client_id in selected_clients_list:
            if client_id not in malicious_clients_list:
                continue

            region_id = client_region_mapping.get(client_id)
            if region_id is None:
                continue

            # Skip if already set for this region this round
            if (region_id in malicious_client.trigger_set_by_region and
                region_id in malicious_client.mask_set_by_region):
                continue

            # Prefer the canonical client's trigger/mask for the region
            canonical_client = canonical_client_for_region.get(region_id, None)

            trig_src = _get_from_store(getattr(malicious_client, "trigger_set", {}), canonical_client)
            mask_src = _get_from_store(getattr(malicious_client, "mask_set", {}), canonical_client)

            # Fallback to any malicious client from this round if canonical not ready
            if trig_src is None or mask_src is None:
                for m_id in malicious_clients_list:
                    trig_src = _get_from_store(malicious_client.trigger_set, m_id)
                    mask_src = _get_from_store(malicious_client.mask_set, m_id)
                    if trig_src is not None and mask_src is not None:
                        logger.warning(f"[TriggerSet] Using fallback trigger/mask from client {m_id} for region {region_id}.")
                        break

            if trig_src is None or mask_src is None:
                logger.warning(f"[TriggerSet] No trigger/mask available for region {region_id} this round.")
                continue

            malicious_client.trigger_set_by_region[region_id] = trig_src
            malicious_client.mask_set_by_region[region_id]    = mask_src

            # tiny fingerprints so you can match later in global eval
            logger.info(f"[TriggerSet] R{region_id} trig_fp={_tiny_fp(trig_src):.3f}, mask_fp={_tiny_fp(mask_src):.3f}")
        

        return (
            weight_accumulator,
            weight_accumulator_by_client,
            aggregated_model_id,
            region_constraints_dict
        )
"""