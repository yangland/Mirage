import copy
import random
import logging
import time
import itertools
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer
from utils.utils import model_dist_norm_var, update_weight_accumulator
from utils.regoin_utils import compute_benign_statistics, build_region_constraints
import random

logger = logging.getLogger("logger")


class No_defense_Server(BasicServer):
    def __init__(self, params, dataloader):
        super(No_defense_Server, self).__init__(params, dataloader)

        # 查看No_defense_Server的所有参数

    # def broadcast_upload_old(self, iteration, benign_client, malicious_client, **kwargs):

    #     logger.info(f"Training on global iteration {iteration} ")

    #     selected_clients_list, malicious_clients_list = self.select_clients(iteration)
    #     ''' 记录当前的训练中，有多少个恶意客户端'''
    #     current_no_of_total_adversaries = 0
    #     for client_id in selected_clients_list:
    #         if client_id in malicious_clients_list:
    #             current_no_of_total_adversaries += 1


    #     weight_accumulator = self.create_weight_accumulator()  # 初始化权重累加器, dict类型
    #     weight_accumulator_by_client = []
    #     update_norm_list = []
    #     global_model_copy = self.create_global_model_copy()
    #     global_model = copy.deepcopy(self.global_model)
    #     aggregated_model_id = [1] * self.params["no_of_participants_per_iteration"]
    #     for client_id in tqdm(selected_clients_list):
    #         if client_id in malicious_clients_list:
    #             client = malicious_client
    #         else:
    #             client = benign_client
    #         client_train_data = self.train_dataloader[client_id]

    #         local_model = copy.deepcopy(self.global_model)

    #         for name, params in local_model.named_parameters():
    #             params.requires_grad = True

    #         local_model.train()
    #         updated_model = client.local_train(iteration, local_model, client_train_data, client_id, test_loader=self.test_dataloader)
    #         update_norm = model_dist_norm_var(updated_model, global_model_copy)  # 计算更新距离全局模型的二范数

    #         update_norm_list.append(round(update_norm.item(), 6))

    #         weight_accumulator, single_wa = update_weight_accumulator(updated_model, copy.deepcopy(self.global_model),
    #                                                                   weight_accumulator)
    #         weight_accumulator_by_client.append(single_wa)
    #         del local_model

    #     for client_ind,client_id in enumerate(selected_clients_list):
    #         logger.info(f"Client {client_id} update norm: {update_norm_list[client_ind]}")
    #     return weight_accumulator, weight_accumulator_by_client, aggregated_model_id


    def broadcast_upload(
        self,
        iteration,
        benign_client,
        malicious_client,
        selected_clients_list,
        malicious_clients_list,
        client_region_mapping,
        **kwargs
        ):
        logger.info(f"Training on global iteration {iteration} ")

        current_no_of_adversaries = sum([1 for client_id in selected_clients_list if client_id in malicious_clients_list])

        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = {}
        update_norm_list = []
        global_model_copy = self.create_global_model_copy()
        global_model = copy.deepcopy(self.global_model)
        aggregated_model_id = [1] * self.params["no_of_participants_per_iteration"]

        # === Step 1: Sample t malicious clients for benign training (for region stat) ===
        benign_sample_num = self.params.get("benign_sample_for_region")
        benign_like_malicious_ids = random.sample(malicious_clients_list, min(benign_sample_num, len(malicious_clients_list)))
        benign_models_from_malicious = []

        for client_id in benign_like_malicious_ids:
            benign_like_model = copy.deepcopy(self.global_model)
            for name, param in benign_like_model.named_parameters():
                param.requires_grad = True
            benign_like_model.train()

            # Benign-style training using benign client logic
            trained_model = benign_client.local_train(iteration, benign_like_model, self.train_dataloader[client_id], client_id)
            benign_models_from_malicious.append(trained_model)

        # === Step 2: Compute region statistics and constraints ===
        region_constraints_dict = None
        if len(benign_models_from_malicious) > 1:
            region_stats = compute_benign_statistics(benign_models_from_malicious, global_model)
            logger.info("[DEBUG] --- Region Statistics Computation ---")
            logger.info(f"[DEBUG] # of benign models: {len(benign_models_from_malicious)}")
            logger.info(f"[DEBUG] Mean pairwise L2 distance: {region_stats['avg_L2_dist']:.4f}")
            logger.info(f"[DEBUG] Mean L2 norm of updates: {region_stats['avg_L2_norm']:.4f}")
            logger.info(f"[DEBUG] Mean cos dist between updates: {region_stats['avg_update_cos_d']:.8f}")
            logger.info(f"[DEBUG] Mean cos dist between weights: {region_stats['avg_weight_cos_d']:.8f}")

            region_constraints_dict = build_region_constraints(
                            stats=region_stats,
                            l2_scale_min=self.params.get("l2_scale_min"),
                            l2_scale_max=self.params.get("l2_scale_max"),
                            cos_scale_min=self.params.get("cos_scale_min"),
                        )
        else:
            logger.warning("Not enough benign-like models to compute region statistics.")
            region_constraints_dict = {i: {} for i in range(8)}  # fallback

        # === Step 3: Use externally provided region assignments ===
        logger.info(f"[Round {iteration}] Using externally provided region assignments: {client_region_mapping}")
        logger.info(f"[Round {iteration}] selected_clients_list: {selected_clients_list}")

        # === Step 4: Train clients ===
        for client_id in tqdm(selected_clients_list):
            client_train_data = self.train_dataloader[client_id]
            local_model = copy.deepcopy(self.global_model)
            for name, param in local_model.named_parameters():
                param.requires_grad = True
            local_model.train()

            if client_id in malicious_clients_list:
                # Get region constraint
                region_id = client_region_mapping.get(client_id)
                updated_model = malicious_client.local_train(
                    iteration, local_model, client_train_data, client_id,
                    test_loader=self.test_dataloader,
                    region_constraints=region_constraints_dict.get(region_id, {}),
                    region_id=region_id
                )
            else:
                updated_model = benign_client.local_train(
                    iteration, local_model, client_train_data, client_id,
                    test_loader=self.test_dataloader
                )

            update_norm = model_dist_norm_var(updated_model, global_model_copy)
            update_norm_list.append(round(update_norm.item(), 6))

            weight_accumulator, single_wa = update_weight_accumulator(
                updated_model, copy.deepcopy(self.global_model), weight_accumulator
            )
            if not isinstance(single_wa, dict):
                print(f"[FATAL] Client {client_id} returned non-dict update: {type(single_wa)}")
                print(f"single_wa: {single_wa}")
                raise RuntimeError("Abort: client update is invalid")
            weight_accumulator_by_client[client_id] = single_wa
            
            if client_id == 0:
                print("[DEBUG] Checking update flow for client 0")
                print("Type of updated_model:", type(updated_model))
                print("Type of global_model:", type(self.global_model))
                print("State dict keys:", list(updated_model.state_dict().keys())[:5])
                print("Update norm:", update_norm)

            if isinstance(single_wa, dict):
                print(f"[DEBUG] Client {client_id} update keys: {list(single_wa.keys())[:5]}")
            else:
                print(f"[ERROR] Client {client_id}'s update is not a dict → it is {type(single_wa)}: {single_wa}")

            del local_model

        for client_ind, client_id in enumerate(selected_clients_list):
            logger.info(f"Client {client_id} update norm: {update_norm_list[client_ind]}")


        # === Step X: Aggregate trigger/mask per region ===
        for client_id in selected_clients_list:
            if client_id not in malicious_clients_list:
                continue

            region_id = client_region_mapping.get(client_id)
            if region_id is None:
                continue

            # Store only the first trigger/mask per region
            if region_id not in malicious_client.trigger_set_by_region:
                malicious_client.trigger_set_by_region[region_id] = malicious_client.trigger_set[client_id]
                malicious_client.mask_set_by_region[region_id] = malicious_client.mask_set[client_id]

        return (
                weight_accumulator,
                weight_accumulator_by_client,
                aggregated_model_id,           
            )

