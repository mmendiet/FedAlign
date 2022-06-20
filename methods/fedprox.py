'''
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
'''

import torch
import logging
from methods.base import Base_Client, Base_Server
import copy
from torch.multiprocessing import current_process

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

    def train(self):
        # train the local model
        self.model.to(self.device)
        global_weight_collector = copy.deepcopy(list(self.model.parameters()))
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                ############
                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((self.args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss = loss + fed_prox_reg
                ########
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
