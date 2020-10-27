# -*- coding:utf-8 -*-
"""
infer on the images captured by the camera
"""
import cv2
from lib.FaceDetector.detector import FaceDetector
from lib.FacialExpressionRecognizer.infer import FerInfer
from lib.FaceRecognizer.infer import FaceInfer

face_detector = FaceDetector()
fer_infer = FerInfer()
face_infer = FaceInfer()

def capturer() :
    cap = cv2.VideoCapture(0)

    idx = 0

    while True :
        ret, frame = cap.read()

        cols = frame.shape[1]
        rows = frame.shape[0]

        detections = face_detector.detect(frame)

        for dete in detections :
            confidence = dete[2]

            xLeftBottom = int(dete[3] * cols)
            yLeftBottom = int(dete[4] * rows)
            xRightTop = int(dete[5] * cols)
            yRightTop = int(dete[6] * rows)

            face = frame[yLeftBottom : yRightTop, xLeftBottom : xRightTop, :]
            # cv2.imwrite("./0001.jpg", face)
            # exit(0)
            fer_porp = fer_infer.infer_image(face)
            _, face_attr = face_infer.infer(face)

            fer = max(fer_porp, key=lambda x : fer_porp[x])

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
            label = "face: %.4f fer: %s %s" % (confidence, fer, face_attr[0])
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow("detections", frame)
        idx = idx + 1

        if cv2.waitKey(100) & 0xff == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    capturer()