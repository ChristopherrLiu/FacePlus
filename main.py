# -*- coding:utf-8 -*-
import time
import logging

from config import *
from utils import *

from lib.FaceRecognizer.train import FaceTrainer
from lib.FacialExpressionRecognizer.train import FerTrainer

def configure_log(conf) :
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(checkdir(conf.logfile_dirname) + "/run_info_{}.log".format(time.strftime("%Y-%m-%d-%H-%M")))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger

if __name__ == "__main__" :
    conf = Config()
    logger = configure_log(conf)
    try :
        if conf.train_object :
            trainer = FerTrainer(conf, logger)
            trainer.train()
        else :
            trainer = FaceTrainer(conf, logger)
            trainer.train()
    except Exception as e :
        logger.error("some errors occur when training.", exc_info=True)