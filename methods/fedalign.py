'''
Code is based on
https://github.com/taoyang1122/GradAug,
https://github.com/taoyang1122/MutualNet.
Also, Lipschitz related functions are from
https://github.com/42Shawn/LONDON/tree/master
'''

import random
import torch
import torch.nn.functional as F

import logging
from methods.base import Base_Client, Base_Server
import torch.nn.functional as F
import models.ComputePostBN as pbn
from torch.multiprocessing import current_process
import numpy as np
import random

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.width_range = client_dict['width_range']
        self.resolutions = client_dict['resolutions']
        self.num_sub = args.num_subnets-1

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
               
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                t_feats, t_out = self.model.extract_feature(images)
                loss = self.criterion(t_out, labels)
                loss.backward()
                loss_CE = loss.item()
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[0]))
                s_feats = self.model.reuse_feature(t_feats[-2].detach())
                
                # Lipschitz loss
                TM_s = torch.bmm(self.transmitting_matrix(s_feats[-2], s_feats[-1]), self.transmitting_matrix(s_feats[-2], s_feats[-1]).transpose(2,1))
                TM_t = torch.bmm(self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()), self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()).transpose(2,1))
                loss = F.mse_loss(self.top_eigenvalue(K=TM_s), self.top_eigenvalue(K=TM_t))
                loss = self.args.mu*(loss_CE/loss.item())*loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

    def transmitting_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def top_eigenvalue(self, K, n_power_iterations=10, dim=1):
        v = torch.ones(K.shape[0], K.shape[1], 1).to(self.device)
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / n

        top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
        return top_eigenvalue

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            ###
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
            self.model = pbn.ComputeBN(self.model, self.train_dataloader, self.resolutions[0], self.device)
            ###
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number)*100
            logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
        return acc

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            ###
            self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
            ###
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number)*100
            logging.info("************* Server Acc = {:.2f} **************".format(acc))
        return acc