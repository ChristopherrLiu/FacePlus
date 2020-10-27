# -*-coding:utf-8-*-
import os.path as osp
import time
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader
import torch.optim as opt
import torch.nn as nn
import numpy as np

import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from .models import MobileFaceNet, ArcMargin
from .verification import verifiy
from .data.face.dataset import FaceDataset
from utils import *

class FaceTrainer(object) :

    def __init__(self, conf, logger) :

        self.conf = conf
        self.device = conf.device

        self.logger = logger

        self.base_path = osp.dirname(osp.abspath(__file__))

        self.net = MobileFaceNet(self.conf.embedding_size).to(self.device)
        self.head = ArcMargin(self.conf.embedding_size, self.conf.face_num_classes).to(self.device)

        self.train_dataset = FaceDataset()
        self.val_dataset = FaceDataset("test")

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.conf.batch_size,
            shuffle=True, pin_memory=True, num_workers=4,
            drop_last=True
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.conf.batch_size,
            shuffle=False
        )

        self.n_optimizer = opt.Adam(self.net.parameters(), lr=self.conf.learning_rate)
        self.h_optimizer = opt.Adam(self.head.parameters(), lr=self.conf.learning_rate)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.t_scheduler = opt.lr_scheduler.MultiStepLR(self.n_optimizer, milestones=self.conf.milestones, gamma=0.2)
        self.iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(self.n_optimizer, self.iter_per_epoch * self.conf.warm)

        checkdir(osp.join(self.base_path, self.conf.visual_dirname))
        checkdir(osp.join(self.base_path, self.conf.ckpt_dirname))
        checkdir(osp.join(self.base_path, self.conf.weight_dirname))


    def evaluate(self, epoch) :
        self.net.eval()
        start = time.time()

        embeddings, labels = list(), list()

        with torch.no_grad() :
            for iter_idx, (img, label) in enumerate(self.val_loader):
                img, label = img.float().to(self.device), label.long().to(self.device)
                embedding = self.net(img)
                embeddings.append(embedding.cpu().numpy())
                labels.append(label.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        tprs, fprs, auc, bacc, bthr = verifiy(embeddings, labels)

        finish = time.time()
        self.logger.info('Evaluating Network.....')
        self.logger.info('Val set: Auc: {:.2%}, Best Accuracy: {:.2%}@threshold={:.2f}, Time consumed:{:.2f}s'.format(
            auc,
            bacc,
            bthr,
            finish - start
        ))

        self.draw_roc_curve(fprs, tprs, epoch)

    def train(self):

        train_infos = dict()
        train_infos['loss'] = list()
        start_epoch = 0

        if self.conf.resume:
            try:
                ckpt_path = get_newest_file(checkdir(osp.join(self.base_path, self.conf.ckpt_dirname)))
                start_epoch, pre_infos = load_train_ckpt([self.net, self.n_optimizer], ckpt_path, self.device)
                train_infos['loss'] += pre_infos
                self.logger.info("resume from {}".format(start_epoch))
            except Exception as e:
                self.logger.error("Fail to load checkpoints...", exc_info=True)

        for epoch in range(start_epoch + 1, self.conf.total_epoch):
            self.net.train()

            start = time.time()
            train_loss, train_corrects = 0, 0

            for iter_idx, (img, label) in enumerate(self.train_loader):
                img, label = img.float().to(self.device), label.long().to(self.device)

                self.h_optimizer.zero_grad()
                self.n_optimizer.zero_grad()
                embedding = self.net(img)
                pred = self.head(embedding, label)
                loss = self.loss_function(pred, label)
                train_loss += loss.item()
                loss.backward()
                predict = nn.functional.softmax(pred, dim=1)
                _, preds = predict.max(1)
                train_corrects += preds.eq(label).sum()
                self.h_optimizer.step()
                self.n_optimizer.step()

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
                self.evaluate(epoch)

            if epoch % 2:
                save_train_ckpt([self.net, self.n_optimizer], epoch, train_infos,
                                osp.join(self.base_path, self.conf.ckpt_dirname, "epoch_{:d}_acc_{:.2%}.ckpt".
                                         format(epoch, float(train_corrects) / len(self.train_loader.dataset))))

        torch.save(self.net.state_dict(),
                   osp.join(self.base_path, self.conf.weight_dirname, "weights.pt"))
        self.draw_loss_curve(train_infos)

    def draw_loss_curve(self, train_infos) :
        plt.close()
        plt.figure()
        color = (random.random(), random.random(), random.random())
        plt.plot([i for i in range(len(train_infos['loss']))], train_infos['loss'], label="train", color=color,
                 marker='.')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Loss Curve", fontsize=14)
        p = osp.join(self.base_path, self.conf.visual_dirname, "loss_curve", "face_loss_{}.jpg".
                     format(time.strftime("%Y-%m-%d-%H-%M")))
        checkdir(os.path.dirname(p))
        plt.savefig(p, dpi=300, bbox_inches='tight', format='jpeg')

    def draw_roc_curve(self, fpr, tpr, epoch):
        color = (random.random(), random.random(), random.random())
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve", fontsize=14)
        plt.plot(fpr, tpr, linewidth=2, color=color, label="epoch{}".format(epoch),
                 marker='.')
        plt.legend()
        p = osp.join(self.base_path, self.conf.visual_dirname, "roc_curve", "roc_epoch-{}_{}.jpg".
                              format(epoch, time.strftime("%Y-%m-%d-%H-%M")))
        checkdir(os.path.dirname(p))
        plt.savefig(p, dpi=300, bbox_inches='tight', format='jpeg')