import cv2 as cv
import mediapipe as mp
import pyautogui as pui
import utility 
import numpy as np
import random


pui.FAILSAFE = False
ScreenWidth,screenHeight = pui.size()
TopX = 150
TopY = 50

BottomX = 50
BottomY = 150


def MoveCursor(Index_finger_Tip,img):
    h,w,c = img.shape
    x = int(Index_finger_Tip.x *w)
    y = int(Index_finger_Tip.y *h)

    x = np.clip(x,TopX,w-BottomX)
    y = np.clip(y,TopY,h-BottomY)

    NewX = np.interp(x,(TopX, w-BottomY),(0,ScreenWidth))
    NewY = np.interp(y,(TopY, h-BottomY),(0,screenHeight))
    
    pui.moveTo(NewX,NewY)

    
    
def DetectGesture(img, Landmark_list,processed):
    #index finger 
    if Landmark_list:
        indexFingerTip = utility.get_FingerTip(processed)
        Thumb_index_Distance = utility.get_distance([Landmark_list[4],Landmark_list[5]])

        IndexFinger_angel = utility.get_angle(Landmark_list[8], Landmark_list[6], Landmark_list[5])
        MiddleFinger_angle = utility.get_angle(Landmark_list[12], Landmark_list[10], Landmark_list[9])


        #Movement of cursor
        if Thumb_index_Distance < 50 and IndexFinger_angel > 90:
            MoveCursor(indexFingerTip,img)

        #Left Click
        if Thumb_index_Distance > 50 and IndexFinger_angel < 50 and MiddleFinger_angle > 90:
            pui.leftClick()
 
        #Right Click
        if Thumb_index_Distance > 50 and IndexFinger_angel > 90 and MiddleFinger_angle < 50:
            pui.rightClick()
        
        #Double Click
        if Thumb_index_Distance > 50 and IndexFinger_angel < 50 and MiddleFinger_angle < 50:
            pui.doubleClick()

        #ScreenShot
        if Thumb_index_Distance < 50 and IndexFinger_angel < 50 and MiddleFinger_angle < 50:
            image = pui.screenshot()
            label = random.randint(1,1000)
            image.save(f"MyScreenshot{label}.png")
            

def main():

    cap = cv.VideoCapture(0)
    mphands = mp.solutions.hands
    hands = mphands.Hands(
        static_image_mode = False,
        model_complexity = 1,
        max_num_hands = 1
    )
    mpdraw = mp.solutions.drawing_utils


    try:
        while cap.isOpened():
            detected , img = cap.read()

            if not detected:
                break
                
            img = cv.flip(img,1)
            h,w,c = img.shape
            #Converting the img from BGR Format to RGB format since mediapipe works on RGB color format
            RGBimg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            processed = hands.process(RGBimg)
            
            Lm_list = []
            #Drawing the connections between the handLandmarks.
            if processed.multi_hand_landmarks:
                hand_landmark = processed.multi_hand_landmarks[0]
                mpdraw.draw_landmarks(img, hand_landmark, mphands.HAND_CONNECTIONS)
                cv.rectangle(img,(TopX,TopY),(w - BottomX, h - BottomY), (255,0,0), 2)

                for lm in hand_landmark.landmark:
                    Lm_list.append((lm.x,lm.y))

            DetectGesture(img,Lm_list,processed)

            cv.imshow("Camera", img)

            if cv.waitKey(1) == 27:
                break

    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()