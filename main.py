#INITIAL SETUP
#----------------------------------------------------------------
import cv2
import os
from cvzone import HandTrackingModule, overlayPNG
import numpy as np

# Read images
intro = cv2.imread(r'CookieCutter-main\frames\img1.jpeg')
kill = cv2.imread(r'CookieCutter-main\frames\img2.png')
winner = cv2.imread(r'CookieCutter-main\frames\img3.png')

# Read camera
cam = cv2.VideoCapture(0)

# Create HandDetector object
detector = HandTrackingModule.HandDetector(maxHands=1, detectionCon=0.77)

# Read images
sqr_img = cv2.imread('CookieCutter-main\img\sqr(2).png')
mlsa = cv2.imread('CookieCutter-main\img\mlsa.png')
while True:
    cv2.imshow('cookie', cv2.resize(intro, (0, 0), fx=0.6, fy=0.6))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#INTRO SCREEN WILL STAY UNTIL Q IS PRESSED

gameOver = False
NotWon =True


brushThickness = 25

drawColor = (255, 0, 255)
imgCanvas = np.zeros((480, 640, 3), np.uint8)
xp, yp = 0, 0

#GAME LOGIC UPTO THE TEAMS
#-----------------------------------------------------------------------------------------
while not gameOver:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    

    hands,img = detector.findHands(img)  # find hands landmarks
    
    camShow = cv2.resize(sqr_img, (0, 0), fx=0.4, fy=0.4)

    camH, camW = camShow.shape[0], camShow.shape[1]
    img[0:camH, -camW:] = camShow
    if hands:
        lmList = hands[0]['lmList']
        if len(lmList) != 0:
            
            # tip of index finger
            x1, y1 = lmList[8][:2]
    
            #Check which fingers are up
            fingers = detector.fingersUp(hands[0])
            
            #Drawing Mode - Index finger is up
            if fingers == [0, 1, 0, 0, 0]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
              
                xp, yp = x1, y1
    
    #zdruzis canvas pa img 
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    

    


    cv2.imshow('cookie',img)

    if cv2.waitKey(1) & 0xFF==ord('q'):  # press 'q' to quit the game
        gameOver = True
#LOSS SCREEN
if NotWon:
    while True:
        cv2.imshow('cookie', cv2.resize(kill, (0, 0), fx=0.5, fy=0.5))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

else:

    cv2.imshow('cookie', cv2.resize(winner, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(125)
    

    while True:
        cv2.imshow('cookie', cv2.resize(winner, (0, 0), fx=0.5, fy=0.5))
        # cv2.imshow('shit',cv2.resize(graphic[3], (0, 0), fx = 0.5, fy = 0.5))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

