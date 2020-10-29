# -*-coding:utf-8-*-
"""
some application Programming Interfaces based on thrift
"""
import os
import os.path as osp
import io
import json
import base64
import cv2
from PIL import Image
import numpy as np

from lib.FaceDetector.detector import FaceDetector
from lib.FacialExpressionRecognizer.infer import FerInfer
from lib.FaceRecognizer.infer import FaceInfer

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from api import userService

class FaceAPI(object) :
    face_detector = FaceDetector()
    fer_infer = FerInfer()
    face_infer = FaceInfer()

    def inferImage(self, image, width, height):

        image = image.replace("data:image/jpeg;base64,", "")

        byte_data = base64.b64decode(image)
        imgio = io.BytesIO(byte_data)
        image = Image.open(imgio)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (224, 224))
        width, height = 224, 224

        res = dict()

        detections = self.face_detector.detect(image)

        for idx, dete in enumerate(detections) :

            x = int(dete[4] * width)
            y = int(dete[3] * height)
            h = int(dete[5] * height) - y
            w = int(dete[6] * width) - x

            face = image[x : x + w, y : y + h, :]
            fer_porp = self.fer_infer.infer_image(face)
            _, face_attr = self.face_infer.infer(face)
            fer = max(fer_porp, key=lambda x: fer_porp[x])

            res[idx] = {
                "loc" : [x * 1. / width, y * 1. / height, w * 1. / width, h * 1. / height],
                "fe" : fer,
                "fe_score" : fer_porp,
                "person_name" : face_attr[0]
            }

        res["total_faces"] = len(detections)

        return json.dumps(res)

if __name__ == "__main__":
    port = 4242
    ip = "127.0.0.1"
    # 创建服务端
    handler = FaceAPI()
    processor = userService.Processor(handler)
    # 监听端口
    transport = TSocket.TServerSocket(ip, port)
    # 选择传输层
    tfactory = TTransport.TBufferedTransportFactory()
    # 选择传输协议
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    # 创建服务端
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    print("start server in python")
    server.serve()
    print("Done")