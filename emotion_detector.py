# 导入必要的包
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

# 加载人脸检测cascade、表情检测CNN权重、构建表情列表
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised",
            "neutral"]

# 如果没有视频，使用电脑自带摄像头
if not args.get("video", False):
    camera = cv2.VideoCapture(0)  # 0表示电脑前置摄像头，1表示电脑后置摄像头
# 否则加载视频
else:
    camera = cv2.VideoCapture(args["video"])

# 迭代
while True:
    # 获取当前帧
    (grabbed, frame) = camera.read()

    # 如果不能获取当前帧，说明视频结束了
    if args.get("video") and not grabbed:
        break

    # 调整当前帧大小，转换成灰度图像
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 初始化画布以进行可视化，然后复制帧以便可以在上面绘制
    canvas = np.zeros((220, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    # 在帧中检测人脸，然后克隆当前帧以便在上面绘制
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # 确保有人脸
    if len(rects) > 0:
        # 确定最大人脸区域
        rect = sorted(rects, reverse=True,
                      key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect

        # 从图像中提取人脸ROI，然后预处理
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 在ROI中进行预测，查找类别标签
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        # 循环标签 + 概率值，并写出
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)

        # 在帧中写出类别
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)

    # 显示分类结果 + 概率值
    cv2.imshow("Face", frameClone)
    cv2.imshow("Probabilities", canvas)  # 用于显示表情的概率分布

    # 按下'q'键结束检测
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()




