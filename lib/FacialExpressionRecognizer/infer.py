# -*-coding:utf-8-*-
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp

import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from .models import FerNet
from config import Config

class FerInfer(object) :
    def __init__(self) :
        self.base_path = osp.dirname(osp.abspath(__file__))
        self.conf = Config()

        self.net = FerNet(self.conf.emotion_num_classes)
        self.net.load_state_dict(torch.load(osp.join(self.base_path, "weights", "weights.pt"), map_location=self.conf.device))
        self.net.eval()
        self.emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

    def infer_image(self, img) :
        img = cv2.resize(img, (42, 42))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img[np.newaxis, :, :]).unsqueeze(0).float()

        with torch.no_grad() :
            pred = self.net(img)
            pred = F.softmax(pred, 1).view(-1)

        res = dict()
        for emotion, score in zip(self.emotions, pred.numpy().tolist()) :
            res[emotion] = score

        return res