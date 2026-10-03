import cv2
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mediapipe as mp
import numpy as np
import pandas as pd
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)

cap = cv2.VideoCapture(0)

data = []
labels = []

gesture_name = input("Enter gesture name (e.g. thumbs_up): ")

print("Press 's' to start collecting, 'q' to quit")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            for handLms in results.hand_landmarks:
                # Draw landmarks for visualization
                for lm in handLms:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                landmarks = []
                for lm in handLms:
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

csv_file = 'gesture_data.csv'

# If file does not exist, write with header=True. If it does, append with header=False
if not os.path.isfile(csv_file):
    df.to_csv(csv_file, mode='w', header=True, index=False)
else:
    df.to_csv(csv_file, mode='a', header=False, index=False)

print(f"Data saved to {csv_file}")
