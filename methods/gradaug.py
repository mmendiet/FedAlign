'''
Code credit to
https://github.com/taoyang1122/GradAug,
https://github.com/taoyang1122/MutualNet,
for implementation of GradAug
'''

import torch
import logging
from methods.base import Base_Client, Base_Server
import numpy as np
import random
import torch.nn.functional as F
import models.ComputePostBN as pbn
from torch.multiprocessing import current_process

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
                # logging.info(images.shape)
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                ####
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                max_output = self.model(images)
                loss = self.criterion(max_output, labels)
                loss.backward()
                max_output_detach = max_output.detach()
                # do other widths and resolution
                width_mult_list = [self.width_range[0]]
                sampled_width = list(np.random.uniform(self.width_range[0], self.width_range[1], self.num_sub))
                width_mult_list.extend(sampled_width)
                for width_mult in sorted(width_mult_list, reverse=True):
                    self.model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    resolution = self.resolutions[random.randint(0, len(self.resolutions)-1)]
                    output = self.model(F.interpolate(images, (resolution, resolution), mode='bilinear', align_corners=True))
                    loss = self.args.mult*torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))
                    loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

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
