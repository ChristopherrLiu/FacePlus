# -*- coding:utf-8 -*-
import os
import os.path as osp
import torch
import logging
import cv2
from PIL import Image
import torchvision.transforms as tfs
import numpy as np

import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from models import MobileFaceNet
from config import Config
from lib.FaceDetector.detector import FaceDetector

base_path = osp.dirname(osp.abspath(__file__))
weight_path = osp.join(base_path, "weights", "weights.pt")
facebank_path = osp.join(base_path, "facebank", "images")
save_path = osp.join(base_path, "facebank", "facebank")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

conf = Config()
net = MobileFaceNet(conf.embedding_size)
net.load_state_dict(torch.load(weight_path, map_location=conf.device))
net.eval()

face_detector = FaceDetector()

img_transform = tfs.Compose([
    tfs.Resize((112, 112)),
    tfs.ToTensor(),
    tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def detect_one_face(path) :
    img_list = os.listdir(path)

    for img_name in img_list:
        img_path = osp.join(path, img_name)

        img = cv2.imread(img_path)
        cols = img.shape[1]
        rows = img.shape[0]
        detections = face_detector.detect(img)

        for dete in detections:
            xLeftBottom = int(dete[3] * cols)
            yLeftBottom = int(dete[4] * rows)
            xRightTop = int(dete[5] * cols)
            yRightTop = int(dete[6] * rows)

            face = img[yLeftBottom: yRightTop, xLeftBottom: xRightTop, :]
            cv2.imwrite(img_path, face)
            break

if __name__ == "__main__" :
    #detect_one_face(osp.join(base_path, "facebank", "images", "Junnan.Liu"))
    person_list = os.listdir(facebank_path)
    imgs, names = list(), list()
    for person in person_list :
        if person == '.gitkeep' or person == 'README.md' : continue
        person_dirname = osp.join(facebank_path, person)
        img_list = os.listdir(person_dirname)

        for img_name in img_list :
            img_path = osp.join(person_dirname, img_name)

            img = Image.open(img_path).convert('RGB')
            img = img_transform(img)
            img = img.unsqueeze(0).contiguous()

            imgs.append(img)
            names.append(person)

            logging.info("load image from {}".format(img_path))

    imgs = torch.cat(imgs, dim=0)

    logging.info("extract features...")
    with torch.no_grad() :
        embeddings = net(imgs)
    embeddings = embeddings.cpu().numpy()

    names = np.array(names)

    np.savez(save_path, embeddings=embeddings, names=names)

    logging.info("done.")