# -*- coding:utf-8 -*-
import os
import os.path as osp
import torch
import cv2
from PIL import Image
import torchvision.transforms as tfs
import numpy as np

import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from .models import MobileFaceNet
from config import Config
from lib.FaceDetector.detector import FaceDetector

class FaceInfer(object) :

    def __init__(self) :
        self.base_path = osp.dirname(osp.abspath(__file__))
        self.conf = Config()

        self.net = MobileFaceNet(self.conf.embedding_size)
        self.net.load_state_dict(torch.load(osp.join(self.base_path, "weights", "weights.pt"), map_location=self.conf.device))
        self.net.eval()

        self.tfs = tfs.Compose([
            tfs.Resize((112, 112)),
            tfs.ToTensor(),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.ori_facebank = np.load(osp.join(self.base_path, "facebank", "facebank.npz"))
        self.face_features = self.ori_facebank['embeddings']
        self.names = self.ori_facebank['names']

    def infer(self, image) :
        if isinstance(image, np.ndarray) :
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.tfs(image)
        if len(image.shape) == 3 :
            image.unsqueeze_(0)

        with torch.no_grad() :
            embedding = self.net(image)
        embedding = embedding.cpu().numpy()

        diff = np.subtract(embedding, self.face_features)
        dist = np.sum(np.square(diff), axis=1, keepdims=True)

        pred, pred_idx = np.min(dist, axis=0), np.argmin(dist, axis=0)
        pred_idx[pred > self.conf.face_threshold] = -1

        res = list()
        for idx in pred_idx :
            if idx == -1 :
                res.append("null")
            else :
                res.append(self.names[idx])

        return pred_idx, res

if __name__ == "__main__" :

    image = Image.open("./facebank/zhaoliying/159116217226.jpg")
    infer = FaceInfer()
    res = infer.infer(image)
    print(res[1])