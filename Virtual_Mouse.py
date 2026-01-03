import cv2
import numpy as np
import mediapipe as mp
import time
import autopy
import math


wCam, hCam = 640, 480
frameR = 100     
smoothening = 5  


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1, 
                      model_complexity=0, 
                      min_detection_confidence=0.7, 
                      min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils

wScr, hScr = autopy.screen.size() 

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        handlm = results.multi_hand_landmarks[0]
        lmList = []
        
        for id, lm in enumerate(handlm.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  
            x2, y2 = lmList[12][1:] 
            
            fingers = []
            # Index Finger
            if lmList[8][2] < lmList[6][2]: fingers.append(1)
            else: fingers.append(0)
            # Middle Finger
            if lmList[12][2] < lmList[10][2]: fingers.append(1)
            else: fingers.append(0)

            if fingers[0] == 1 and fingers[1] == 0:
                
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                
                try:
                    autopy.mouse.move(clocX, clocY)
                except ValueError:
                    pass 
                
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            if fingers[0] == 1 and fingers[1] == 1:
                length = math.hypot(x2 - x1, y2 - y1)
                
                
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                
                if length < 40:
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED) 
                    autopy.mouse.click()
                    time.sleep(0.1) 

        
        mpdraw.draw_landmarks(img, handlm, mphands.HAND_CONNECTIONS)

    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
