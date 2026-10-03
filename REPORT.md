# 🖐 Gesture Recognition & Automation System - Project Report

## 1. Executive Summary
The **Gesture Recognition System** is a real-time computer vision application that translates hand gestures into system-level commands. By leveraging MediaPipe for landmark detection and a custom-trained machine learning model, the system provides a touchless interface for controlling media, scrolling through documents, and capturing screenshots.

---

## 2. Technical Architecture

### 2.1 Core Technologies
- **MediaPipe Hands**: Used for high-fidelity 21-point hand landmark extraction.
- **OpenCV**: Handles video capture, frame preprocessing, and real-time visualization.
- **Scikit-learn**: Powering the gesture classification model (trained on landmark coordinate data).
- **CustomTkinter**: A modern, dark-themed dashboard for system monitoring and control.
- **PyAutoGUI**: Facilitates the automation of system-level keyboard and mouse actions.

### 2.2 Functional Flow
1. **Input**: Webcam stream is captured at ~30-60 FPS.
2. **Processing**: MediaPipe detects hands and extracts 3D coordinates (x, y, z) for 21 landmarks.
3. **Inference**: The normalized coordinates are fed into a pre-trained `gesture_model.pkl` to identify the gesture.
4. **Execution**: The identified gesture triggers a mapped system action (e.g., Spacebar for Palm).
5. **Visualization**: The dashboard updates real-time logs, statistics, and confidence indicators.

---

## 3. Supported Gestures & Automation

| Gesture | Icon | Action Triggered | Typical Use Case |
| :--- | :---: | :--- | :--- |
| **Palm** | 🖐 | `Space` | Play/Pause video or music |
| **Index Finger** | ☝ | `Volume Mute` | Instant audio muting |
| **Right Slide** | 👉 | `Scroll Up` | Navigating web pages/documents |
| **Left Slide** | 👈 | `Scroll Down` | Navigating web pages/documents |
| **Screenshot** | 📸 | `Save File` | Instant screen capture to disk |

---

## 4. User Interface Features

The system features a premium **Dark Mode Dashboard** designed for usability:
- **Real-time Camera Feed**: Overlays skeletal hand landmarks and detected labels.
- **Gesture Badge**: Visual confirmation of the currently recognized gesture.
- **Action Log**: A timestamped history of all automated actions performed.
- **Gesture Statistics**: Live tracking of gesture frequencies with progress bars.
- **Control Panel**: Adjustable cooldown sliders and an automation toggle to prevent unintended triggers.

---

## 5. Development Cycle

### Phase 1: Data Collection
Using `data-collector.py`, a custom dataset (`gesture_data.csv`) was created by capturing landmark coordinates for each gesture class in various orientations and lighting conditions.

### Phase 2: Model Training
The `model.py` script utilizes the gathered data to train a classification model. The resulting model is serialized into `gesture_model.pkl` for low-latency inference.

### Phase 3: Integration & UI
The core logic was integrated into `ui_app.py`, focusing on a responsive multi-threaded architecture to ensure the camera feed remains smooth while processing inference and automation.

---

## 6. Performance Indicators
- **High Accuracy**: Robust recognition of complex gestures (slides) using relative landmark positions.
- **Low Latency**: Inference time is optimized for real-time feedback loop.
- **System Safety**: Integrated cooldown logic ensures single gestures don't trigger multiple cascading actions.

---

## 7. Future Enhancements
- **Multi-Hand Support**: Enabling different actions for left and right hands.
- **Custom Mapping**: Allow users to remap gestures to different keyboard shortcuts via the UI.
- **Depth-Based Actions**: Using the Z-axis (hand distance) for volume or brightness control.
