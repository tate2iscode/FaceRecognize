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

#얼굴 감지후 얼굴 데이터 리턴
def detectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img.copy()

def mp_detectFace(img, face_detection):
    i = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i.flags.writeable = False
    result = face_detection.process(i)
    if result.detections:
        result_img = img.copy()
        for detection in result.detections:
            mp_drawing.draw_detection(result_img, detection)
            #print(detection)
            #print("---------------------------")
        #print("+++++++++++++++++++")
    else:
        result_img = img.copy()
    return result_img

def mp_detectFace_v2(img, face_detection, width, height):
    i = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i.flags.writeable = False
    result = face_detection.process(i)
    if result.detections:
        result_img = img.copy()
        for detection in result.detections:
            x = int(detection.location_data.relative_bounding_box.xmin*width) - 100
            y = int(detection.location_data.relative_bounding_box.ymin*height) - 100
            w = int(detection.location_data.relative_bounding_box.width*width) + 200
            h = int(detection.location_data.relative_bounding_box.height*height) + 200
            #print(x, y)
            #cv2.imwrite("facetest/p2/"+str(n)+".jpg", img[y:y+h, x:x+w].copy())
            #print("p2/"+str(n)+".jpg")
            #n = n + 1
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    else:
        result_img = img.copy()
    return result_img

def return_face_pos(img, face_detection, width, height):
    i = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i.flags.writeable = False
    result_pos = []
    result = face_detection.process(i)
    if result.detections:
        for detection in result.detections:
            r = {}
            x = int(detection.location_data.relative_bounding_box.xmin*width)
            y = int(detection.location_data.relative_bounding_box.ymin*height)
            w = int(detection.location_data.relative_bounding_box.width*width)
            h = int(detection.location_data.relative_bounding_box.height*height)
            r["x"] = x
            r["y"] = y
            r["w"] = w
            r["h"] = h
            result_pos.append(r)
    return result_pos

def face_embedding(img, pos):
    p1 = img[pos["y"]:pos["h"] + pos["y"], pos["x"]:pos["x"] + pos["w"]].copy()
    #cv2.imshow("p1", p1)

    p2 = face_align.align(534, p1, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if p2 is not None:
        p2 = cv2.resize(p2, (150, 150))
        face_descriptor = facerec.compute_face_descriptor(p2)
        return np.array(face_descriptor)
    else:
        return None

def mp_detectFace_v3(img, face_detection, width, height, facedata):
    i = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i.flags.writeable = False
    result = face_detection.process(i)
    if result.detections:
        result_img = img.copy()
        for detection in result.detections:
            x = int(detection.location_data.relative_bounding_box.xmin*width) - 100
            y = int(detection.location_data.relative_bounding_box.ymin*height) - 100
            w = int(detection.location_data.relative_bounding_box.width*width) + 200
            h = int(detection.location_data.relative_bounding_box.height*height) + 200
            input_pos = {"x": x, "y": y, "w": w, "h": h}
            vector = face_embedding(img, input_pos)
            if vector is not None:
                distance = compare(facedata, vector)
            else:
                distance = "No"
            print(distance)

            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    else:
        result_img = img.copy()
    return result_img

def compare(v1, v2):
    v1 = v1 - v2
    v1 = v1 ** 2
    v1 = v1.sum()
    v1 = v1 * 1/2
    return v1