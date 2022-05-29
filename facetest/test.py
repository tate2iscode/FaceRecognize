import dlib
import cv2
import numpy as np
import openface
from tool import *
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

print(dlib.__version__)

p_model = "models/shape_predictor_68_face_landmarks.dat"

face_align = openface.AlignDlib(p_model)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    path = "training-images/test2.jpg"
    img = cv2.imread(path)
    pos = return_face_pos(img, face_detection, img.shape[1], img.shape[0])
    print(pos)

    #p1 = img[pos[0]["y"]:pos[0]["h"]+pos[0]["y"], pos[0]["x"]:pos[0]["x"]+pos[0]["w"]].copy()
    #cv2.imshow("p1", p1)

    #p2 = face_align.align(534, p1, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    #cv2.imshow("p2", p2)
    #cv2.waitKey()
    #p2 = cv2.resize(p2, (150, 150))
    print(face_embedding(img, pos))