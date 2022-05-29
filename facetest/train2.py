import cv2
import numpy as np
import mediapipe as mp
import openface
import dlib

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

p_model = "models/shape_predictor_68_face_landmarks.dat"

face_align = openface.AlignDlib(p_model)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

