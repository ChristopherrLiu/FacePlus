# -*- coding:utf-8 -*-
import numpy as np
from sklearn import metrics

def calculate_roc(thresholds:np.ndarray, similarity:np.ndarray, actual_issame:np.ndarray) :
    """
    to calculate ROC curve
    largely inspired by insightFace
    :param thresholds:an array of different thresholds
    :param similarity: Matrix of feature's similarity
    :param actual_issame: labels
    :return:
    """
    _ones = np.ones(similarity.size)
    _zeros = np.zeros(similarity.size)
    tprs = np.zeros(thresholds.size)
    fprs = np.zeros(thresholds.size)
    accs = np.zeros(thresholds.size)
    for threshold_idx, threshold in enumerate(thresholds) :
        predict_issame = np.less(similarity, threshold)

        # calculate accuracy
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tprs[threshold_idx] = 0 if (tp + fn == 0) else tp.astype(np.float) / (tp + fn).astype(np.float)
        fprs[threshold_idx] = 0 if (fp + tn == 0) else fp.astype(np.float) / (fp + tn).astype(np.float)
        accs[threshold_idx] = (tp + tn).astype(np.float) / predict_issame.shape[0]

    auc = metrics.auc(fprs, tprs)
    best_acc = np.max(accs)
    best_threshold_idx = np.argmax(accs)
    best_threshold = thresholds[best_threshold_idx]

    return tprs, fprs, auc, best_acc, best_threshold

def verifiy(features:np.ndarray, labels:np.ndarray) :
    """
    to verifiy model performance
    :param features: features extracted from the model [N * embedding_size]
    :param labels: labels of the input images [N * 1]
    :return:
    """
    similarity = np.sum((features[0::2] - features[1::2]) ** 2, 1)
    thresholds = np.arange(0, 4, 0.01)
    actual_issame = np.equal(labels[0::2], labels[1::2])

    return calculate_roc(thresholds, similarity, actual_issame)