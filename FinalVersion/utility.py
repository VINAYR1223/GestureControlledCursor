import numpy as np
import mediapipe as mp

import os
import sys
sys.path.append(os.path.abspath("utility.py"))

def get_FingerTip(processed):
    if processed.multi_hand_landmarks:
        handLandmark = processed.multi_hand_landmarks[0]
        return handLandmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

def get_angle(a,b,c ):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return
    
    x1,y1 = landmark_list[0]
    x2,y2 = landmark_list[1]

    distance = np.hypot(x2-x1,y2-y1)
    L = np.interp(distance,[0,1],[1,1000])
    return L
    