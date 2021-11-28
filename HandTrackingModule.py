import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

class handDetector():
    def __init__(self,mode = False,maxHands= 2, detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon) # this class only uses RGB img
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # self.results yapınca her yerde her methoddakullanabiliyoz
        # print(results.multi_hand_landmarks) # elimizi koyduğu muzda bir sürü değer verir
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
               if draw:
                self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS) # orjinal resmi oynatıcaz çizdiricez tek el için elimizin noktalarını belirliyoruz,3. method ile handLms nokta görüntüsünü alıyoruz
        return img
    def findPosition(self,img,handNo=0,draw=True):
                lmList = []
                if self.results.multi_hand_landmarks:
                    myHand = self.results.multi_hand_landmarks[handNo]
                    for id,lm in enumerate(myHand.landmark):
                        # print(id,lm)
                        h,w,c = img.shape
                        cx,cy = int(lm.x*w), int(lm.y*h)
                        # print(id,cx,cy)
                        lmList.append([id,cx,cy])
                        if draw:
                            cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
                        # if id ==8:
                        #     cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
                return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        # lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,('FPS:'+str(int(fps))),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2) # fps i buraya text olarak yazdırdık

        cv2.imshow('Image',img)
        if cv2.waitKey(20) & 0xFF ==27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

