# -*-coding:utf-8-*-
import os.path as osp
import time
import logging
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader
import torch.optim as opt
import torch.nn as nn

import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from .models import FerNet
from .data.fer2013.dataset import FerDataset
from utils import *

class FerTrainer(object) :

    def __init__(self, conf, logger) :

        self.conf = conf
        self.device = conf.device

        self.logger = logger

        self.base_path = osp.dirname(osp.abspath(__file__))

        self.net = FerNet(self.conf.emotion_num_classes).to(self.device)

        self.train_dataset = FerDataset()
        self.val_dataset = FerDataset("test")

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.conf.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.conf.batch_size,
            shuffle=False
        )

        self.optimizer = opt.Adam(self.net.parameters(), lr=self.conf.learning_rate)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.t_scheduler = opt.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.conf.milestones, gamma=0.2)
        self.iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.iter_per_epoch * self.conf.warm)

        checkdir(osp.join(self.base_path, self.conf.visual_dirname))
        checkdir(osp.join(self.base_path, self.conf.ckpt_dirname))
        checkdir(osp.join(self.base_path, self.conf.weight_dirname))

    def evaluate(self) :

        self.net.eval()
        start = time.time()
        test_corrects, test_loss = 0, 0

        for iter_idx, (img, label) in enumerate(self.val_loader):
            img, label = img.to(self.device).float(), label.to(self.device).long()
            predict = self.net(img)
            loss = self.loss_function(predict, label)
            test_loss += loss.item()
            predict = nn.functional.softmax(predict, dim=1)
            _, preds = predict.max(1)
            test_corrects += preds.eq(label).sum()

        finish = time.time()
        self.logger.info('Evaluating Network.....')
        self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {:.2%}, Time consumed:{:.2f}s'.format(
            test_loss / len(self.val_loader.dataset),
            float(test_corrects) / len(self.val_loader.dataset),
            finish - start
        ))

    def train(self):

        train_infos = dict()
        train_infos['loss'] = list()
        start_epoch = 0

        if self.conf.resume:
            try:
                ckpt_path = get_newest_file(osp.join(self.base_path, self.conf.ckpt_dirname))
                start_epoch, pre_infos = load_train_ckpt([self.net, self.optimizer], ckpt_path, self.device)
                train_infos['loss'] += pre_infos
                self.logger.info("resume from {}".format(start_epoch))
            except Exception as e:
                self.logger.error("Fail to load checkpoints...", exc_info=True)

        for epoch in range(start_epoch + 1, self.conf.total_epoch):
            self.net.train()

            start = time.time()
            train_loss, train_corrects = 0, 0

            for iter_idx, (img, label) in enumerate(self.train_loader):
                img, label = img.to(self.device).float(), label.to(self.device).long()

                self.optimizer.zero_grad()
                predict = self.net(img)
                loss = self.loss_function(predict, label)
                train_loss += loss.item()
                loss.backward()
                predict = nn.functional.softmax(predict, dim=1)
                _, preds = predict.max(1)
                train_corrects += preds.eq(label).sum()
                self.optimizer.step()

                if epoch <= self.conf.warm:
                    self.warmup_scheduler.step()

            if epoch > self.conf.warm:
                self.t_scheduler.step()

            finish = time.time()
            train_infos['loss'].append(train_loss / len(self.train_loader.dataset))
            self.logger.info('epoch {} : Average loss: {:.4f}, Accuracy: {:.2%}, Time consumed:{:.2f}s'.format(
                epoch,
                train_loss / len(self.train_loader.dataset),
                float(train_corrects) / len(self.train_loader.dataset),
                finish - start
            ))

            if epoch % 2:
                self.evaluate()

            if epoch % 2:
                save_train_ckpt([self.net, self.optimizer], epoch, train_infos,
                                osp.join(self.base_path, self.conf.ckpt_dirname, "epoch_{:d}_acc_{:.2%}.ckpt".format(epoch, float(
                                    train_corrects) / len(self.train_loader.dataset))))

        torch.save(self.net.state_dict(),
                   osp.join(self.base_path, self.conf.weight_dirname, "weights.pt")
                   )

        self.draw_loss_curve(train_infos)

    def draw_loss_curve(self, train_infos) :
        color = (random.random(), random.random(), random.random())
        plt.plot([i for i in range(len(train_infos['loss']))], train_infos['loss'], label="train", color=color,
                 marker='.')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Loss Curve", fontsize=14)
        p = osp.join(self.base_path, self.conf.visual_dirname, "loss_curve",
                              "fer_loss_{}.jpg".format(time.strftime("%Y-%m-%d-%H-%M")))
        checkdir(os.path.dirname(p))
        plt.savefig(p, dpi=300, bbox_inches='tight', format='jpeg')