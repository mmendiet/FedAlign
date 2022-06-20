'''
Code credit to https://github.com/QinbinLi/MOON
for implementation of thier method, MOON.
'''

import torch
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, projection=True)
        self.prev_model = self.model_type(self.num_classes, KD=True, projection=True)
        # self.prev_model.load_state_dict(self.model.state_dict())
        self.global_model = self.model_type(self.num_classes, KD=True, projection=True)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temp = 0.5

    def run(self, received_info):
        client_results = []
        self.global_model.load_state_dict(received_info['global'])
        for client_idx in self.client_map[self.round]:
            self.prev_model.load_state_dict(received_info['prev'][client_idx])
            self.load_client_state_dict(received_info['global'])
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader)*self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append({'weights':weights, 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()
        self.round += 1
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.global_model.to(self.device)
        self.prev_model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                #####
                pro1, out = self.model(x)
                pro2, _ = self.global_model(x)

                posi = self.cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                pro3, _ = self.prev_model(x)
                nega = self.cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= self.temp
                labels = torch.zeros(x.size(0)).to(self.device).long()

                loss2 = self.args.mu * self.criterion(logits, labels)

                loss1 = self.criterion(out, target)
                loss = loss1 + loss2
                #####
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                    epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        # self.prev_model.load_state_dict(weights)
        return weights

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                _, pred = self.model(x)
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
        self.model = self.model_type(self.num_classes, KD=True, projection=True)
        self.prev_models = {x:self.model.cpu().state_dict() for x in range(self.args.client_number)}
    
    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        for x in received_info:
            self.prev_models[x['client_index']] = x['weights']
        server_outputs = [{'global':g, 'prev':self.prev_models} for g in server_outputs]
        return server_outputs

    def start(self):
        return [{'global':self.model.cpu().state_dict(), 'prev':self.prev_models} for x in range(self.args.thread_number)]
    
    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                _, pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number)*100
            logging.info("************* Server Acc = {:.2f} **************".format(acc))
        return acc