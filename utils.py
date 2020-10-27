# -*- coding:utf-8 -*-
import torch
from torch.optim.lr_scheduler import _LRScheduler
import os
import os.path as osp

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def checkdir(path) :
    """
    may make directory
    :param path: path of directory
    :return: path
    """
    if not osp.exists(path) :
        os.makedirs(path)
    return path

def get_newest_file(dir) :
    """
    get the newest file name
    :param dir: path of directory
    :return: path
    """
    lists = os.listdir(dir)
    lists.sort(key=lambda fn:os.path.getmtime(dir + "\\" + fn))
    file_new = os.path.join(dir, lists[-1])
    return file_new

class WarmUpLR(_LRScheduler):
    """
    warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def save_train_ckpt(modules_optims, epoch, train_infos, ckpt_file) :
    """
    Save state_dict's of modules/optimizers to file during training
    :param modules_optims: A list, which members are either torch.nn.optimizer
    or torch.nn.Module.
    :param ep: the current epoch number
    :param ckpt_file: The file path.
    """
    state_dicts = [m.state_dict() for m in modules_optims]
    ckpt = dict(state_dicts=state_dicts, epoch=epoch, train_infos=train_infos)
    checkdir(os.path.dirname(os.path.abspath(ckpt_file)))
    torch.save(ckpt, ckpt_file)

def load_train_ckpt(modules_optims, ckpt_file, device):
    """
    Load state_dict's of modules/optimizers from file.
    :param modules_optims: A list, which members are either torch.nn.optimizer
      or torch.nn.Module.
    :param ckpt_file: The file path.
    :param load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers
      to cpu type.
    """
    ckpt = torch.load(ckpt_file, map_location=device)
    for m, sd in zip(modules_optims, ckpt['state_dicts']):
        if m:
            m.load_state_dict(sd)
    return ckpt['epoch'], ckpt['train_infos']
