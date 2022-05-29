import dlib
import cv2
import numpy as np
import openface
import os
from tool import *

facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    path = "p2/"
    filelist = os.listdir(path)
    filelist.remove(".DS_Store")
    print(filelist)

    result = []
    for name in filelist:
        print(name)
        img = cv2.imread(path+name)
        pos = return_face_pos(img, face_detection, img.shape[1], img.shape[0])
        print(pos)
        if len(pos) != 0:
            for p in pos:
                p1 = img[p["y"]:p["h"] + p["y"], p["x"]:p["x"] + p["w"]].copy()
                cv2.imwrite("p1/" + name, p1)
                result.append(face_embedding(img, p))
                print("추가")

    result = np.array(result)
    np.save("p1/obama.npy", result)
