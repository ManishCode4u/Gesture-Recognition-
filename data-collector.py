import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data = []
labels = []

gesture_name = input("Enter gesture name (e.g. thumbs_up): ")

print("Press 's' to start collecting, 'q' to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if cv2.waitKey(1) & 0xFF == ord('s'):
                data.append(landmarks)
                labels.append(gesture_name)
                print(f"Captured: {len(data)} samples")

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('gesture_data.csv', mode='a', header=False, index=False)

print("Data saved to gesture_data.csv")
