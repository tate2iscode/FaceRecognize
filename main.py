from tool import *
import numpy as np
import cv2
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

def main():
    width = 640
    height = 480
    facedata = np.load("facetest/myfacedata.npy")
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #n = 0
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cv2.waitKey(33) < 0:
            ret, frame = capture.read()
            #detectFaceImg = detectFace(frame.copy())
            #mp_detectFaceImg = mp_detectFace(frame.copy(), face_detection)
            mp_detectFaceImg_v3 = mp_detectFace_v4(frame.copy(), face_detection, width, height, facedata)
            #n = n + 1


            # 1 opencv와 mediapipe 차이
            #show_result = np.hstack((detectFaceImg, mp_detectFaceImg))


            show_result = mp_detectFaceImg_v3


            cv2.imshow("video capture", show_result)
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
