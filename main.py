import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0)
# hand detection model by mediapipe
mpHands = mp.solutions.hands
# method details
hands = mpHands.Hands()

# draw hand landmarks, and its style settings
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(255,255,255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(255,0,0), thickness=3)

# FPS
# previous-time
pTime = 0
# current-time
cTime = 0


while True:
    ret, frame = cap.read()

    if ret:
        # all frames needs to convert to RGB (opencv(BGR)->mediapipe(RGB))
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # processed result
        result = hands.process(imgRGB)
        # hand landmarks coordinate
        # print(result.multi_hand_landmarks)

        # get the height and width size of every frame
        imgHeight = frame.shape[0]
        imgWidth = frame.shape[1]


        # if landmarks coordinate detected
        if result.multi_hand_landmarks:
            
            # draw landmark
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                
                # draw landmark id
                for i, lm in enumerate(handLms.landmark):
                    # landmark position coordinate setting
                    xPosition = np.int(lm.x * imgWidth)
                    yPosition = np.int(lm.y * imgHeight)
                    print(i, xPosition, yPosition)
                    # put id string next to the landmark
                    cv2.putText(frame, str(i), (xPosition-30,yPosition+5), cv2.FONT_ITALIC, 0.3, (255,255,255), 1)
    
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (30,50), cv2.FONT_ITALIC, 0.5, (128,128,128), 1)


        cv2.imshow('Capture', frame)

    if cv2.waitKey(20) == ord('q'):
        break