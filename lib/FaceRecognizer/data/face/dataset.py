# -*- coding:utf-8 -*-
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image

import lmdb
import pickle

dataset_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class FaceDataset(Dataset) :
    def __init__(self, mode="train") :
        super(FaceDataset, self).__init__()

        self.base_path = osp.dirname(osp.abspath(__file__))

        self.db_path = osp.join(self.base_path, "transformed/{}.lmdb".format(mode))
        self.env = lmdb.open(
            self.db_path,
            readonly=True, lock=False,
            readahead=False, meminit=False
        )

        with self.env.begin(write=False) as txn :
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))


    def __len__(self) :
        return self.length

    def __getitem__(self, index) :

        with self.env.begin(write=False) as txn :
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        img = unpacked[0]
        img = Image.fromarray(img)
        img = dataset_tfs(img)
        img.permute((2, 0, 1))

        label = unpacked[1]

        return img, label