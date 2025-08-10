import copy
import random
import logging
import time
import itertools
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer
from utils.utils import model_dist_norm_var, update_weight_accumulator, model_weight_diff, update_weight_accumulator_direct
from utils.regoin_utils import compute_benign_statistics, build_region_constraints
import random

logger = logging.getLogger("logger")


class No_defense_Server(BasicServer):
    def __init__(self, params, dataloader, full_train_dataset=None):
        super(No_defense_Server, self).__init__(params, dataloader, full_train_dataset)

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
        **kwargs
    ):
        logger.info(f"Training on global iteration {iteration} ")

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

        # === Step X: Aggregate trigger/mask per region ===
        for client_id in selected_clients_list:
            if client_id not in malicious_clients_list:
                continue

            region_id = client_region_mapping.get(client_id)
            if region_id is None:
                continue

            if region_id not in malicious_client.trigger_set_by_region:
                canonical_client = canonical_client_for_region[region_id]
                malicious_client.trigger_set_by_region[region_id] = malicious_client.trigger_set[canonical_client]
                malicious_client.mask_set_by_region[region_id] = malicious_client.mask_set[canonical_client]

        return (
            weight_accumulator,
            weight_accumulator_by_client,
            aggregated_model_id,
            region_constraints_dict
        )
