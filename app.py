from datetime import datetime
import cv2
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time

# 1. Load the trained model
print("Loading model...")
model = joblib.load('gesture_model.pkl')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)

cap = cv2.VideoCapture(0)

# Variables to control the speed of actions
last_action_time = 0
cooldown = 1.0  # Seconds to wait before triggering the action again

print("System Ready. Press 'q' to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = landmarker.detect(mp_image)

        gesture_name = "None"

        if results.hand_landmarks:
            for handLms in results.hand_landmarks:
                
                # Draw circles on landmarks for visualization
                for lm in handLms:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Extract landmarks (same logic as data collector)
                landmarks = []
                for lm in handLms:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Predict the gesture
                # We reshape because the model expects a 2D array
                prediction = model.predict([landmarks])
                gesture_name = prediction[0]

                # --- AUTOMATION SECTION ---
                # EDIT THIS PART to match your specific gesture names!
                current_time = time.time()
                if current_time - last_action_time > cooldown:

                    if gesture_name == "palm" and (current_time - last_action_time > cooldown):
                        # Example: Press Spacebar (Play/Pause)
                        pyautogui.press('space')
                        print("Action: Space (Play/Pause)")
                        last_action_time = current_time

                    elif gesture_name == "index" and (current_time - last_action_time > cooldown):
                        # Example: Mute Volume
                        pyautogui.press('volumemute')
                        print("Action: Toggled Mute/Unmute")
                        last_action_time = current_time

                    elif gesture_name == "right slide":
                        pyautogui.scroll(-300)
                        print("action: scrolling up")
                        last_action_time = current_time

                    elif gesture_name == "left slide":
                        pyautogui.scroll(300)
                        print("action: scrolling down")
                        last_action_time = current_time

                    elif gesture_name == "screenshot":
                        filename = f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                        pyautogui.screenshot(filename)
                        print(f'action: screenshot saved at {filename}')
                        last_action_time=current_time

        # Display gesture on screen
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Automation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
