# -*- coding:utf-8 -*-
import os.path as osp
from cv2 import dnn

BASE_DIR = osp.abspath(osp.dirname(__file__))


class FaceDetector(object) :
    """
    detect face using DNN in opencv
    """
    def __init__(self) :
        self.inWidth = 300
        self.inHeight = 300
        self.confThreshold = 0.5

        self.prototxt = osp.join(BASE_DIR, 'weights/deploy.prototxt')
        self.caffemodel = osp.join(BASE_DIR, 'weights/res10_300x300_ssd_iter_140000.caffemodel')
        self.net = dnn.readNetFromCaffe(self.prototxt, self.caffemodel)

    def detect(self, frame):
        """
        detect face on image
        :param frame: image in opencv format
        :return: face information array in shape of [N * 7]
        """

        self.net.setInput(dnn.blobFromImage(frame, 1.0, (self.inWidth, self.inHeight), (104.0, 177.0, 123.0), False, False))
        detections = self.net.forward()

        detections = detections[detections[:, :, :, 2] > 0.5]

        perf_stats = self.net.getPerfProfile()

        return detections