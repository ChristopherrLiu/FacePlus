# -*-coding:utf-8-*-
import os
import logging
import numpy as np
import csv
import lmdb
import pickle
import pyarrow as pa

resource_path = "resource/fer2013.csv"
save_dirname = "transformed"

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

rec = ((0, 42, 0, 42), (6, 48, 0, 42), (0, 42, 6, 48), (6, 48, 6, 48), (3, 45, 3, 45))

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)

if __name__ == "__main__" :
    train_imgs, train_ids = list(), list()
    test_imgs, test_ids = list(), list()
    count = 0
    with open(resource_path) as csv_file :
        reader = csv.DictReader(csv_file)
        for row in reader :
            try :
                emotion = row['emotion']
                pixels = row['pixels']
                usage = row['Usage']

                data_array = list(map(float, pixels.split()))
                img = np.array(data_array).reshape(48, 48)

                if usage == "Training":
                    train_ids.append(int(emotion))
                    train_imgs.append(img)

                else:
                    test_ids.append(int(emotion))
                    test_imgs.append(img)

                logging.info("load {} images".format(count))
            except Exception as e :
                logging.error("fails at no.{} image, beacuse: {}".format(count, e))

            count = count + 1

    #np.savez(save_dirname + "/train", ids=train_ids, imgs=train_imgs)
    #np.savez(save_dirname + "/test", ids=test_ids, imgs=test_imgs)

    lmdb_path = save_dirname + "/train.lmdb"
    if not os.path.exists(save_dirname):
        os.mkdir(save_dirname)
    logging.info("generate LMDB to {}".format(lmdb_path))
    db = lmdb.open(
        lmdb_path,
        map_size=925910970, readonly=False,
        meminit=False, map_async=True
    )

    txn = db.begin(write=True)

    for idx, (image, label) in enumerate(zip(train_imgs, train_ids)):
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % 1000 == 0:
            logging.info("[{}/{}]".format(idx, len(train_ids)))
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(train_ids))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    logging.info("flushing database...")
    db.sync()
    db.close()
    logging.info("done.")

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

    for idx, (image, label) in enumerate(zip(test_imgs, test_ids)):
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % 1000 == 0:
            logging.info("[{}/{}]".format(idx, len(test_ids)))
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(test_ids))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    logging.info("flushing database...")
    db.sync()
    db.close()

    logging.info("done")