import copy

import torch

from participants.clients.BasicClient import BasicClient
from utils.utils import model_dist_norm_var
from utils.visualize import visualize_batch
from utils.utils import poisoned_batch_injection
class BenignClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader):
        super(BenignClient, self).__init__(params, train_dataloader, test_dataloader)

    def local_train(self, iteration, model, train_loader, client_id, **kwargs):
        '''
        客户端本地训练，在本地数据上进行训练，并返回更新后的模型
        :param iteration:
        :param model:
        :param train_loader:
        :param client_id:
        :return:
        '''

        lr = self.get_lr(iteration)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.params['poisoned_momentum'],
                                    weight_decay=self.params['poisoned_weight_decay'])

        for epoch in range(self.params["benign_retrain_no_times"]):
            # pred_cache = torch.tensor([], device=self.params['run_device'])
            for batch_idx, batch in enumerate(train_loader):
                inputs, labels = batch
                inputs, labels = inputs.to(self.params["run_device"]), labels.to(self.params["run_device"])
                outputs = model(inputs)
                # pred = torch.argmax(outputs, dim=1)
                # pred_cache = torch.cat((pred_cache, pred), dim=0)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        #     if epoch % 10 == 0:
        #         correct = 0
        #         total = 0
        #         loss_total = 0
        #         model.eval()
        #         loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
        #         with torch.no_grad():
        #             for batch_idx, batch in enumerate(train_loader):
        #                 inputs, labels = batch
        #                 inputs, labels = inputs.to(self.params["run_device"]), labels.to(self.params["run_device"])
        #                 outputs = model(inputs)
        #                 pred = outputs.argmax(dim=1, keepdim=True)
        #                 correct += pred.eq(labels.view_as(pred)).sum().item()
        #                 total += labels.size(0)
        #                 loss = loss_func(outputs, labels)
        #                 loss_total += loss.item()
        #         acc = correct / total
        #         print(f"Client {client_id} local train acc: {acc:.4f}, loss: {loss_total:.4f}")
        # raise


        '''返回模型更新'''
        model.eval()
        return model


