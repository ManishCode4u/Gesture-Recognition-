# 🖐 Gesture AI — Real-Time Hand Gesture Recognition & Automation System

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Demo-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://gesture-recognition-system.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-00f2fe?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-KNN%20Classifier-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An intelligent, cross-platform **Human-Computer Interaction (HCI)** system that recognizes hand gestures in real-time to automate operating system tasks and browser interactions. Features both an **Edge AI Web Application** (ready for instant Vercel deployment) and a **Native Python Desktop Suite** (with PyAutoGUI OS automation and CustomTkinter dark UI).

---

## 🚀 Live Demo & Web App (Vercel Ready)

The web edition runs **100% in-browser** using MediaPipe WebAssembly and WebGL:
- ⚡ **Zero-latency** edge processing — no video stream leaves your device.
- 🎮 **Interactive Action Studio**: Gesture-controlled Media Player, Page Feed Scroller, and Live Snapshot Capturer.
- 📊 **Real-time 21 Landmark 3D Inspector**: Live joint coordinates, confidence meters, and event logs.
- 🔊 **Web Audio Synthesizer**: Subtle sci-fi sound cues for action execution.

---

## 🎯 Gesture Automation Mappings

| Gesture Icon | Gesture Name | Trigger Condition | Web App Action | Desktop OS Action (Python) |
| :---: | :--- | :--- | :--- | :--- |
| 🖐 | **Palm Open** | All 5 fingers extended | Play / Pause Media Player | Press `Spacebar` (Play/Pause) |
| ☝ | **Index Point** | Index up, others folded | Toggle Audio Mute / Unmute | Press `Volume Mute` |
| 👉 | **Right Slide** | Swipe hand rightward | Scroll Feed Down | Mouse Scroll Down (`-300`) |
| 👈 | **Left Slide** | Swipe hand leftward | Scroll Feed Up | Mouse Scroll Up (`+300`) |
| 📸 | **Screenshot / Pinch** | Thumb & Index tip touch | Capture Canvas Snapshot | Save high-res OS Screenshot |
| 👍 | **Thumbs Up** | Thumb up, 4 fingers folded | Trigger Confetti Celebration | Confirm / Like Action |

---

## 📂 Project Architecture

```
gesture-recognition/
├── index.html              # Modern Cyberpunk Web App UI
├── style.css               # Glassmorphism design system & neon styling
├── app.js                  # In-browser MediaPipe AI vision & gesture dispatcher
├── vercel.json             # Vercel deployment configuration & security headers
├── ui_app.py               # CustomTkinter Dark UI desktop application
├── app.py                  # OpenCV + PyAutoGUI desktop automation engine
├── data-collector.py       # Landmark dataset recording utility
├── model.py                # KNN gesture model training pipeline
├── gesture_model.pkl       # Pre-trained machine learning model
├── hand_landmarker.task    # Google MediaPipe hand landmark model
├── gesture_data.csv        # Hand landmark dataset (63 features per sample)
└── requirements.txt        # Python desktop dependencies
```

---

## 💻 Local Desktop Setup (Python)

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

```bash
# Clone the repository
git clone https://github.com/ManishCode4u/Gesture-Recognition-.git
cd Gesture-Recognition-

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Desktop Applications

**Modern Dark GUI Application:**
```bash
python ui_app.py
```

**Headless Automation Script:**
```bash
python app.py
```
*Press `q` on the webcam window to safely exit.*

---

## 🧠 Model Training & Custom Gestures

Want to add your own custom gestures?

1. **Collect Data**:
   ```bash
   python data-collector.py
   ```
   *Enter gesture name when prompted, press `s` to capture frames, and `q` when finished.*

2. **Train Model**:
   ```bash
   python model.py
   ```
   *Trains a K-Nearest Neighbors (`k=5`) classifier on 63 landmark coordinates and exports `gesture_model.pkl`.*

---

## 🌐 Deploy to Vercel in 1-Click

1. Fork or push this repository to your GitHub account: `https://github.com/ManishCode4u/Gesture-Recognition-`
2. Go to [Vercel Dashboard](https://vercel.com/new).
3. Select **Import Git Repository** and choose `Gesture-Recognition-`.
4. Click **Deploy** (Zero configuration needed — `vercel.json` handles everything).

---

## 🛡️ Privacy & Security
This project is built with privacy-first principles. Camera frames are processed entirely on-device (client-side in browser via WebAssembly / locally in Python). No webcam streams or personal data are ever transmitted to external servers.

---

## 👨‍💻 Author & Contributions
Developed with ❤️ by **[Manish Kumar](https://github.com/ManishCode4u)**.  
Contributions, issues, and feature requests are welcome! Feel free to star ⭐ the repository.
