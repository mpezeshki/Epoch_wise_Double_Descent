import torch
import numpy as np


class ModelWrapper(object):

    def __init__(self, model, optimizer, criterion, device, only_read_out, reg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.only_read_out = only_read_out
        self.reg = reg

    def train_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        feats = self.model(inputs)
        if self.only_read_out:
            feats = feats.detach()
        outputs = self.model.fc(feats)
        if outputs.shape[1] == 1:
            targets = 2.0 * (targets > 4).float() - 1.0
            loss = torch.log(1 + torch.exp(-targets * outputs[:, 0])).mean()
            
            Q = (targets * outputs[:, 0]).var()
            loss += self.reg * Q

            loss.backward()
            self.optimizer.step()
            correct = (torch.sign(outputs[:, 0]) == targets).sum().item()
            acc = correct / targets.size(0)
            return loss.item(), acc, correct

        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0)
        # return loss.item(), acc, correct
        return loss.item(), acc, correct, Q.item()

    def eval_all(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        indivs = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                loss, correct, indiv = self.eval_on_batch(inputs, targets)
                indivs += [indiv]
                total += targets.size(0)
                test_loss += loss
                test_correct += correct
            test_loss /= (batch_idx+1)
            test_acc = test_correct / total
        return test_loss, test_acc, torch.cat(indivs)

    def eval_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        feats = self.model(inputs)
        outputs = self.model.fc(feats)
        if outputs.shape[1] == 1:
            targets = 2.0 * (targets > 4).float() - 1.0
            loss = torch.log(1 + torch.exp(-targets * outputs[:, 0])).mean()
            correct = (torch.sign(outputs[:, 0]) == targets).sum().item()
            return loss.item(), correct, (torch.sign(outputs[:, 0]) != targets).int()

        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct, 1.0 - predicted.eq(targets).int()

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()



