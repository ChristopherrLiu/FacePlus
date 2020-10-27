# -*- coding:utf-8 -*-
import argparse
import torch
import os.path as osp

from utils import *

class Config() :
    parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
    parser.add_argument("--resume", "-r", type=str2bool, default='t', help='resume from checkpoint')
    parser.add_argument("--train_object", "-to", type=str2bool, default='t',
                        help='select training model. if True train Facial Expression Recognizer else Face Recognizer')

    parser.add_argument("--batch_size", "-b", type=int, default=32, help='the size of each batch')
    parser.add_argument("--total_epoch", "-te", type=int, default=50, help='the total epochs of training')
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help='the learning rate when training')
    args = parser.parse_args()

    BASE_DIR = osp.dirname(osp.abspath(__file__))

    visual_dirname = "visual"
    ckpt_dirname = "checkpoints"
    weight_dirname = "weights"
    logfile_dirname = "log"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    total_epoch = args.total_epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    warm = 1
    milestones = [int(total_epoch * 0.6), int(total_epoch * 0.7), int(total_epoch * 0.9)]

    embedding_size = 512

    face_num_classes = 8001
    emotion_num_classes = 7

    face_threshold = 1.4

    emotions = {
        '0': 'anger',  # 生气
        '1': 'disgust',  # 厌恶
        '2': 'fear',  # 恐惧
        '3': 'happy',  # 开心
        '4': 'sad',  # 伤心
        '5': 'surprised',  # 惊讶
        '6': 'normal',  # 中性
    }

    resume = args.resume
    train_object = args.train_object

