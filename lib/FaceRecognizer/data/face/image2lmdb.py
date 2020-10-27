# -*-coding:utf-8-*-
import os
import logging
import random
import numpy as np
from PIL import Image

import lmdb
import pickle

train_set = "train.txt"
test_set = "test.txt"

save_dirname = "transformed"
resource_dirname = "resource"

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)

if __name__ == "__main__" :

    if not os.path.exists(save_dirname) :
        os.mkdir(save_dirname)

    # tranform training dataset
    timgs, tids, tnames = list(), list(), list()

    train_list = []
    try :
        train_list = open(train_set, "r")
        logging.info("transform training dataset...")
    except Exception as e :
        logging.error("fail to load training file.")

    true_id = -1
    pre_id = -1
    for iter in train_list :
        info = str(iter)
        path = info.split(" ")[0]
        id = info.split(" ")[1]
        img_name = path.split("/")[1]
        img_path = resource_dirname + "/" + path

        if int(id) != pre_id :
            true_id = true_id + 1
            if true_id > 8000 :
                break
        try :
            img = np.array(Image.open(img_path).resize((112, 112)))

            timgs.append(img)
            tids.append(true_id)
            tnames.append(img_name)

            logging.info("success to load {}.".format(img_path))

        except Exception as e :
            logging.error("fail to load {} because of {}.".format(img_path, e))

        pre_id = int(id)

    #timgs, tids, tnames = np.array(timgs), np.array(tids), np.array(tnames)

    lmdb_path = save_dirname + "/train.lmdb"
    if not os.path.exists(save_dirname) :
        os.mkdir(save_dirname)
    logging.info("generate LMDB to {}".format(lmdb_path))
    db = lmdb.open(
        lmdb_path,
        map_size=4954218394, readonly=False,
        meminit=False, map_async=True
    )

    txn = db.begin(write=True)

    for idx, (image, label) in enumerate(zip(timgs, tids)) :
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % 1000 == 0 :
            logging.info("[{}/{}]".format(idx, len(tnames)))
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(tnames))]
    with db.begin(write=True) as txn :
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    logging.info("flushing database...")
    db.sync()
    db.close()
    #np.savez(save_dirname + "/train", imgs=timgs, ids=tids, names=tnames)
    logging.info("done.")
    exit()

    # tranform test dataset
    timgs, tids, tnames = list(), list(), list()

    test_list, ids, paths = [], [], []
    tag = dict()
    try:
        test_list = open(test_set, "r")
        logging.info("transform test dataset...")
    except Exception as e:
        logging.error("fail to load test file.")

    for iter in test_list:
        info = str(iter)
        path = info.split(" ")[0]
        id = info.split(" ")[1]
        ids.append(id)
        paths.append(path)
        if id not in tag.keys() :
            tag[id] = 1
        else :
            tag[id] = tag[id] + 1

    two_imgs, one_imgs = [], []
    for k, v in tag.items() :
        if v == 2 :
            two_imgs.append(k)
        elif v == 1 :
            one_imgs.append(k)

    select_ids = random.sample(two_imgs, 3000)
    pre_id = -1

    for (id, path) in zip(ids, paths) :
        if id not in select_ids : continue
        img_name = path.split("/")[1]
        img_path = resource_dirname + "/" + path
        try:
            img = np.array(Image.open(img_path).resize(((112, 112))))

            timgs.append(img)
            tids.append(int(id))
            tnames.append(img_name)

            logging.info("success to load {}.".format(img_path))

        except Exception as e:
            logging.error("fail to load {} because of {}.".format(img_path, e))

    count = 0
    for (id, path) in zip(ids, paths) :
        if id in select_ids :
            continue
        if id == pre_id :
            continue
        img_name = path.split("/")[1]
        img_path = resource_dirname + "/" + path
        try:
            img = np.array(Image.open(img_path).resize(((112, 112))))

            timgs.append(img)
            tids.append(int(id))
            tnames.append(img_name)

            logging.info("success to load {}.".format(img_path))

        except Exception as e:
            logging.error("fail to load {} because of {}.".format(img_path, e))

        pre_id = id
        count = count + 1
        if count == 6000:
            break

    #timgs, tids, tnames = np.array(timgs), np.array(tids), np.array(tnames)
    lmdb_path = save_dirname + "/test.lmdb"
    if not os.path.exists(save_dirname):
        os.mkdir(save_dirname)
    logging.info("generate LMDB to {}".format(lmdb_path))
    db = lmdb.open(
        lmdb_path,
        map_size=652304742, readonly=False,
        meminit=False, map_async=True
    )

    txn = db.begin(write=True)

    for idx, (image, label) in enumerate(zip(timgs, tids)):
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % 1000 == 0:
            logging.info("[{}/{}]".format(idx, len(tnames)))
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(tnames))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    logging.info("flushing database...")
    db.sync()
    db.close()
    #np.savez(save_dirname + "/test", imgs=timgs, ids=tids, names=tnames)
    logging.info("done.")