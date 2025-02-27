import copy
import torch
import logging
import time
from torch import nn
import numpy as np
from participants.clients.BasicClient import BasicClient
from utils.utils import poisoned_batch_injection

logger = logging.getLogger("logger")


class MaliciousClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader):
        super(MaliciousClient, self).__init__(params, train_dataloader, test_dataloader)
        self.init_trigger_mask()

    def local_train(self, iteration, model, train_loader, client_id, **kwargs):
        '''
        Malicious client 的训练过程，包括trigger植入

        :param iteration:
        :param model:
        :param train_loader:
        :param client_id:
        :return:
        '''
        cache_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(cache_model.parameters(), lr=self.params["poisoned_lr"],
                                    momentum=self.params['poisoned_momentum'],
                                    weight_decay=self.params['poisoned_weight_decay'])
        ce_loss = nn.functional.cross_entropy
        # optimizer = torch.optim.Adam(cache_model.parameters(), lr=0.01)

        # 获取当前时间
        current_time = time.time()
        for epoch in range(self.params["poisoned_retrain_no_times"]):
            for batch_idx, batch in enumerate(train_loader):

                inputs, labels = batch
                inputs, labels = poisoned_batch_injection(batch, trigger=self.trigger_set[client_id],
                                                          mask=self.mask_set[client_id], is_eval=False,
                                                          label_swap=self.params["poison_label_swap"][client_id])
                inputs, labels = inputs.to(self.params["run_device"]), labels.to(self.params["run_device"])
                optimizer.zero_grad()
                outputs = cache_model(inputs)
                loss = ce_loss(outputs, labels, label_smoothing=0.001)
                loss.backward()
                optimizer.step()
        return cache_model
