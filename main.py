import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if ret:

        cv2.imshow('Capture', frame)

    if cv2.waitKey(20) == ord('q'):
        break