import copy
import random
import logging
import time

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

    def broadcast_upload_old(self, iteration, benign_client, malicious_client, **kwargs):

        logger.info(f"Training on global iteration {iteration} ")

        selected_clients_list, malicious_clients_list = self.select_clients(iteration)
        ''' 记录当前的训练中，有多少个恶意客户端'''
        current_no_of_adversaries = 0
        for client_id in selected_clients_list:
            if client_id in malicious_clients_list:
                current_no_of_adversaries += 1


        weight_accumulator = self.create_weight_accumulator()  # 初始化权重累加器, dict类型
        weight_accumulator_by_client = []
        update_norm_list = []
        global_model_copy = self.create_global_model_copy()
        global_model = copy.deepcopy(self.global_model)
        aggregated_model_id = [1] * self.params["no_of_participants_per_iteration"]
        for client_id in tqdm(selected_clients_list):
            if client_id in malicious_clients_list:
                client = malicious_client
            else:
                client = benign_client
            client_train_data = self.train_dataloader[client_id]

            local_model = copy.deepcopy(self.global_model)

            for name, params in local_model.named_parameters():
                params.requires_grad = True

            local_model.train()
            updated_model = client.local_train(iteration, local_model, client_train_data, client_id, test_loader=self.test_dataloader)
            update_norm = model_dist_norm_var(updated_model, global_model_copy)  # 计算更新距离全局模型的二范数

            update_norm_list.append(round(update_norm.item(), 6))

            weight_accumulator, single_wa = update_weight_accumulator(updated_model, copy.deepcopy(self.global_model),
                                                                      weight_accumulator)
            weight_accumulator_by_client.append(single_wa)
            del local_model

        for client_ind,client_id in enumerate(selected_clients_list):
            logger.info(f"Client {client_id} update norm: {update_norm_list[client_ind]}")
        return weight_accumulator, weight_accumulator_by_client, aggregated_model_id


    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        logger.info(f"Training on global iteration {iteration} ")

        selected_clients_list, malicious_clients_list = self.select_clients(iteration)
        current_no_of_adversaries = sum([1 for client_id in selected_clients_list if client_id in malicious_clients_list])

        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = []
        update_norm_list = []
        global_model_copy = self.create_global_model_copy()
        global_model = copy.deepcopy(self.global_model)
        aggregated_model_id = [1] * self.params["no_of_participants_per_iteration"]

        # === Step 1: Sample t malicious clients for benign training (for region stat) ===
        benign_sample_num = self.params.get("benign_sample_for_region", 10)
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
        region_constraints = None
        if len(benign_models_from_malicious) > 1:
            stats = compute_benign_statistics(benign_models_from_malicious, global_model)
            region_constraints = build_region_constraints(stats)
        else:
            logger.warning("Not enough benign-like models to compute region statistics.")
            region_constraints = {i: {} for i in range(8)}  # fallback

        # === Step 3: Assign region IDs to malicious clients ===
        region_assignments = {}
        malicious_clients_this_round = [cid for cid in selected_clients_list if cid in malicious_clients_list]
        possible_region_ids = list(region_constraints.keys())

        for client_id in malicious_clients_this_round:
            region_assignments[client_id] = random.choice(possible_region_ids)

        # === Step 4: Train clients ===
        for client_id in tqdm(selected_clients_list):
            client_train_data = self.train_dataloader[client_id]
            local_model = copy.deepcopy(self.global_model)
            for name, param in local_model.named_parameters():
                param.requires_grad = True
            local_model.train()

            if client_id in malicious_clients_list:
                # Get region constraint
                region_id = region_assignments.get(client_id, 0)
                constraint = region_constraints[region_id]
                updated_model = malicious_client.local_train(
                    iteration, local_model, client_train_data, client_id,
                    test_loader=self.test_dataloader,
                    region_constraint=constraint
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
            weight_accumulator_by_client.append(single_wa)
            del local_model

        for client_ind, client_id in enumerate(selected_clients_list):
            logger.info(f"Client {client_id} update norm: {update_norm_list[client_ind]}")

        return weight_accumulator, weight_accumulator_by_client, aggregated_model_id
