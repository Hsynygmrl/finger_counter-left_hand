import cv2
import numpy as np
import mediapipe as mp
import os
import time 
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4,8,12,16,20]
# folderPath = 'FingerImages'
# myList = os.listdir(folderPath)
# print(myList)
# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}')
#     # w_ratio = 0.5
#     # h_ratio = 0.5
#     image = cv2.resize(image,(200,100)) # yarı oranda küçülttük
#     # print(image.shape)
#     overlayList.append(image)
# print(len(overlayList))
while True:
    success, img = cap.read()
    img= detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]: # baş parmak için
                fingers.append(1)
        else:
                fingers.append(0)
        # for other fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        # if totalFingers != 0:
        #     img[0:100,0:200] = overlayList[totalFingers-1] 
        # else:
        #     cv2.putText(img,'Buraya bak',(0,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(450,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()