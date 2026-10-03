
import os
os.environ["PYTHONUTF8"] = "1"

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import cv2
import threading
import queue
import time
import joblib
import numpy as np
from datetime import datetime
import mediapipe as mp
import pyautogui


#  Setup

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


#  Colour palette

BG_DARK     = "#0d1117"
BG_CARD     = "#161b22"
BG_CARD2    = "#1c2332"
ACCENT      = "#58a6ff"
ACCENT2     = "#1f6feb"
ACCENT_GLOW = "#388bfd"
SUCCESS     = "#3fb950"
WARNING     = "#e3b341"
DANGER      = "#f85149"
TEXT_PRI    = "#f0f6fc"
TEXT_SEC    = "#8b949e"
BORDER      = "#30363d"

GESTURE_ICONS = {
    "palm":       ("🖐",  WARNING),
    "index":      ("☝",  ACCENT),
    "right slide":("👉",  SUCCESS),
    "left slide": ("👈",  SUCCESS),
    "screenshot": ("📸",  "#a371f7"),
    "None":       ("✋",  TEXT_SEC),
}


#  Helper widgets

class GlowLabel(ctk.CTkLabel):
    """A label that simulates a glow effect by padding + fg colour."""
    pass


class PulsingDot(ctk.CTkCanvas):
    """Animated pulsing status dot."""
    def __init__(self, master, color=SUCCESS, size=14, **kw):
        super().__init__(master, width=size, height=size,
                         bg=BG_CARD, highlightthickness=0, **kw)
        self.color = color
        self.size  = size
        self._alpha = 1.0
        self._dir   = -1
        self._draw()

    def _draw(self):
        self.delete("all")
        r = max(2, int(self.size / 2 * self._alpha))
        c = self.size // 2
        self.create_oval(c - r, c - r, c + r, c + r,
                         fill=self.color, outline="")
        self._alpha += self._dir * 0.04
        if self._alpha <= 0.4 or self._alpha >= 1.0:
            self._dir *= -1
        self.after(60, self._draw)

    def set_color(self, color):
        self.color = color


class ScrolledLog(ctk.CTkScrollableFrame):
    """Scrollable log panel."""
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._rows = []

    def add_entry(self, icon, message, color=TEXT_PRI):
        ts = datetime.now().strftime("%H:%M:%S")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=2)

        ctk.CTkLabel(row, text=ts, font=("JetBrains Mono", 10),
                     text_color=TEXT_SEC, width=65, anchor="w"
                     ).pack(side="left", padx=(4, 0))
        ctk.CTkLabel(row, text=icon, font=("Segoe UI Emoji", 12),
                     width=24, anchor="w"
                     ).pack(side="left", padx=2)
        ctk.CTkLabel(row, text=message, font=("Inter", 11),
                     text_color=color, anchor="w"
                     ).pack(side="left", padx=2, fill="x", expand=True)

        self._rows.append(row)
        if len(self._rows) > 80:
            self._rows[0].destroy()
            self._rows.pop(0)

        self._parent_canvas.yview_moveto(1.0)



#  Main Application

class GestureUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("✨ Gesture Recognition System")
        self.geometry("1380x820")
        self.minsize(1100, 680)
        self.configure(fg_color=BG_DARK)

        # State 
        self.running        = False
        self.automation_on  = ctk.BooleanVar(value=True)
        self.cooldown_var   = ctk.DoubleVar(value=1.0)
        self.current_gesture= "None"
        self.last_action_t  = 0
        self.gesture_counts = {"palm": 0, "index": 0, "right slide": 0,
                               "left slide": 0, "screenshot": 0}

        self.frame_queue    = queue.Queue(maxsize=2)
        self._cap           = None
        self._thread        = None

        # Model + MediaPipe
        self._model_loaded = False
        self._mp_ready     = False
        self._load_model_async()

        # Build UI
        self._build_ui()
        self._poll_frame()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    
    #  Model / MediaPipe loading
    
    def _load_model_async(self):
        def _load():
            try:
                self._model = joblib.load("gesture_model.pkl")
                self._model_loaded = True
            except Exception as e:
                self._model = None
                print(f"[WARN] Model load failed: {e}")

            try:
                BaseOptions        = mp.tasks.BaseOptions
                HandLandmarker     = mp.tasks.vision.HandLandmarker
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                VisionRunningMode  = mp.tasks.vision.RunningMode
                opts = HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path="hand_landmarker.task"),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=1)
                self._landmarker = HandLandmarker.create_from_options(opts)
                self._mp_ready   = True
            except Exception as e:
                self._landmarker = None
                print(f"[WARN] MediaPipe load failed: {e}")

        threading.Thread(target=_load, daemon=True).start()

    
    #  UI Construction
    
    def _build_ui(self):
        # Top header
        self._build_header()

        # Main content area
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=18, pady=(6, 14))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        left  = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")

        self._build_camera_panel(left)
        self._build_right_panel(right)

    # Header
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=BG_CARD,
                           corner_radius=0, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Logo area
        logo_f = ctk.CTkFrame(hdr, fg_color="transparent")
        logo_f.pack(side="left", padx=20)

        ctk.CTkLabel(logo_f, text="🖐", font=("Segoe UI Emoji", 28)
                     ).pack(side="left", padx=(0, 8))
        title_f = ctk.CTkFrame(logo_f, fg_color="transparent")
        title_f.pack(side="left")
        ctk.CTkLabel(title_f, text="Gesture Control",
                     font=("Inter", 20, "bold"),
                     text_color=TEXT_PRI
                     ).pack(anchor="w")
        ctk.CTkLabel(title_f, text="Real-time gesture recognition system",
                     font=("Inter", 11),
                     text_color=TEXT_SEC
                     ).pack(anchor="w")

        # Status right side
        status_f = ctk.CTkFrame(hdr, fg_color="transparent")
        status_f.pack(side="right", padx=20)

        self._status_dot = PulsingDot(status_f, color=TEXT_SEC)
        self._status_dot.pack(side="left", padx=(0, 6))
        self._status_label = ctk.CTkLabel(
            status_f, text="Idle",
            font=("Inter", 12), text_color=TEXT_SEC)
        self._status_label.pack(side="left")

        # Divider line
        ctk.CTkFrame(self, fg_color=BORDER, height=1,
                     corner_radius=0).pack(fill="x")

    # Camera Panel
    def _build_camera_panel(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Current gesture badge (above camera)
        badge_row = ctk.CTkFrame(parent, fg_color="transparent")
        badge_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self._gesture_badge = ctk.CTkFrame(
            badge_row, fg_color=BG_CARD2,
            corner_radius=12, border_width=1,
            border_color=BORDER)
        self._gesture_badge.pack(side="left")

        self._gesture_icon_lbl = ctk.CTkLabel(
            self._gesture_badge, text="✋",
            font=("Segoe UI Emoji", 32))
        self._gesture_icon_lbl.pack(side="left", padx=(14, 6), pady=8)

        gest_text = ctk.CTkFrame(self._gesture_badge, fg_color="transparent")
        gest_text.pack(side="left", padx=(0, 16))
        ctk.CTkLabel(gest_text, text="DETECTED GESTURE",
                     font=("Inter", 9, "bold"),
                     text_color=TEXT_SEC).pack(anchor="w")
        self._gesture_name_lbl = ctk.CTkLabel(
            gest_text, text="None",
            font=("Inter", 22, "bold"),
            text_color=TEXT_SEC)
        self._gesture_name_lbl.pack(anchor="w")

        # Confidence placeholder
        self._conf_bar = ctk.CTkProgressBar(
            badge_row, width=160, height=8,
            corner_radius=4,
            fg_color=BORDER, progress_color=ACCENT)
        self._conf_bar.pack(side="right", padx=6)
        self._conf_bar.set(0)
        ctk.CTkLabel(badge_row, text="Confidence",
                     font=("Inter", 10), text_color=TEXT_SEC
                     ).pack(side="right")

        # Camera feed
        cam_frame = ctk.CTkFrame(parent, fg_color=BG_CARD,
                                 corner_radius=14,
                                 border_width=1, border_color=BORDER)
        cam_frame.grid(row=1, column=0, sticky="nsew")

        self._cam_label = ctk.CTkLabel(cam_frame, text="")
        self._cam_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Placeholder while camera is off
        self._make_placeholder()

        # Controls below camera
        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ctrl.columnconfigure(0, weight=1)

        self._start_btn = ctk.CTkButton(
            ctrl, text="▶  Start Camera",
            font=("Inter", 13, "bold"),
            fg_color=ACCENT2, hover_color=ACCENT,
            corner_radius=10, height=42,
            command=self._toggle_camera)
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        clr_btn = ctk.CTkButton(
            ctrl, text="🗑 Clear Log",
            font=("Inter", 12),
            fg_color=BG_CARD2, hover_color=BORDER,
            text_color=TEXT_SEC, corner_radius=10, height=42,
            width=120,
            command=self._clear_log)
        clr_btn.grid(row=0, column=1)

    # Right panel
    def _build_right_panel(self, parent):
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        # Controls card
        ctrl_card = self._card(parent, "⚙️  Controls")
        ctrl_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(ctrl_card, text="Automation",
                     font=("Inter", 12), text_color=TEXT_SEC
                     ).pack(side="left", padx=(14, 8))
        self._auto_switch = ctk.CTkSwitch(
            ctrl_card, text="",
            variable=self.automation_on,
            onvalue=True, offvalue=False,
            fg_color=BORDER, progress_color=SUCCESS,
            button_color=TEXT_PRI)
        self._auto_switch.pack(side="left")

        ctk.CTkLabel(ctrl_card, text="Cooldown",
                     font=("Inter", 12), text_color=TEXT_SEC
                     ).pack(side="left", padx=(20, 8))
        self._cooldown_slider = ctk.CTkSlider(
            ctrl_card, from_=0.3, to=3.0,
            variable=self.cooldown_var,
            width=120, button_color=ACCENT,
            progress_color=ACCENT2, fg_color=BORDER,
            command=self._on_cooldown_change)
        self._cooldown_slider.pack(side="left")
        self._cooldown_val_lbl = ctk.CTkLabel(
            ctrl_card, text="1.0s",
            font=("Inter", 12, "bold"),
            text_color=ACCENT, width=40)
        self._cooldown_val_lbl.pack(side="left", padx=4)

        # Gesture stats card
        stats_card = self._card(parent, "📊  Gesture Statistics")
        stats_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        stats_card.configure(fg_color=BG_CARD, corner_radius=12)

        self._stat_labels = {}
        gestures = [("palm","🖐"), ("index","☝"), ("right slide","👉"),
                    ("left slide","👈"), ("screenshot","📸")]
        for g, icon in gestures:
            row = ctk.CTkFrame(stats_card, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=3)

            ctk.CTkLabel(row, text=icon,
                         font=("Segoe UI Emoji", 14), width=20
                         ).pack(side="left")
            ctk.CTkLabel(row, text=g.title(),
                         font=("Inter", 11), text_color=TEXT_PRI,
                         anchor="w", width=90
                         ).pack(side="left", padx=6)

            bar = ctk.CTkProgressBar(row, height=8, corner_radius=4,
                                     fg_color=BORDER,
                                     progress_color=GESTURE_ICONS.get(g, ("", ACCENT))[1])
            bar.set(0)
            bar.pack(side="left", fill="x", expand=True, padx=6)

            cnt = ctk.CTkLabel(row, text="0",
                               font=("JetBrains Mono", 11, "bold"),
                               text_color=TEXT_SEC, width=28)
            cnt.pack(side="right")
            self._stat_labels[g] = (bar, cnt)

        # --- Action log ---
        log_header = ctk.CTkFrame(parent, fg_color="transparent")
        log_header.grid(row=2, column=0, sticky="nsew")
        log_header.rowconfigure(1, weight=1)
        log_header.columnconfigure(0, weight=1)

        hdr_row = ctk.CTkFrame(log_header, fg_color=BG_CARD2,
                               corner_radius=10)
        hdr_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(hdr_row, text="📋  Action Log",
                     font=("Inter", 13, "bold"),
                     text_color=TEXT_PRI
                     ).pack(side="left", padx=14, pady=8)
        self._log_count_lbl = ctk.CTkLabel(
            hdr_row, text="0 events",
            font=("Inter", 10), text_color=TEXT_SEC)
        self._log_count_lbl.pack(side="right", padx=14)

        self._log_panel = ScrolledLog(
            log_header,
            fg_color=BG_CARD, corner_radius=10,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT2)
        self._log_panel.grid(row=1, column=0, sticky="nsew")

    # Helper builders
    def _card(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD2,
                             corner_radius=12,
                             border_width=1, border_color=BORDER,
                             height=52)
        frame.pack_propagate(False)
        ctk.CTkLabel(frame, text=title,
                     font=("Inter", 12, "bold"),
                     text_color=TEXT_PRI
                     ).pack(side="left", padx=14)
        return frame

    def _make_placeholder(self):
        """Black canvas with centered text as a camera placeholder."""
        w, h = 620, 440
        img = Image.new("RGB", (w, h), color=(14, 18, 28))
        draw = ImageDraw.Draw(img)
        draw.text((w // 2 - 90, h // 2 - 12),
                  "📷  Camera Offline",
                  fill=(80, 90, 110))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                               size=(w, h))
        self._cam_label.configure(image=ctk_img, text="")
        self._cam_label._image_ref = ctk_img

    
    #  Camera / Processing thread
    
    def _toggle_camera(self):
        if self.running:
            self.running = False
            self._start_btn.configure(text="▶  Start Camera",
                                      fg_color=ACCENT2)
            self._set_status("Idle", TEXT_SEC)
        else:
            self.running = True
            self._start_btn.configure(text="⏹  Stop Camera",
                                      fg_color=DANGER)
            self._set_status("Running", SUCCESS)
            self._thread = threading.Thread(
                target=self._process_loop, daemon=True)
            self._thread.start()

    def _process_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._log("⚠️", "Camera not found!", DANGER)
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gesture = "None"

            if self._mp_ready and self._model_loaded:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=rgb)
                results  = self._landmarker.detect(mp_image)

                if results.hand_landmarks:
                    for handLms in results.hand_landmarks:
                        # Draw landmark dots
                        for lm in handLms:
                            x = int(lm.x * frame.shape[1])
                            y = int(lm.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5,
                                       (88, 166, 255), -1)
                            cv2.circle(frame, (x, y), 7,
                                       (31, 111, 235), 1)

                        # Draw connections between landmarks (hardcoded MediaPipe hand connections)
                        HAND_CONNECTIONS = [
                            (0,1),(1,2),(2,3),(3,4),       # Thumb
                            (0,5),(5,6),(6,7),(7,8),       # Index
                            (5,9),(9,10),(10,11),(11,12),  # Middle
                            (9,13),(13,14),(14,15),(15,16),# Ring
                            (13,17),(17,18),(18,19),(19,20),# Pinky
                            (0,17)
                        ]
                        for conn in HAND_CONNECTIONS:
                            lm_a = handLms[conn[0]]
                            lm_b = handLms[conn[1]]
                            xa = int(lm_a.x * frame.shape[1])
                            ya = int(lm_a.y * frame.shape[0])
                            xb = int(lm_b.x * frame.shape[1])
                            yb = int(lm_b.y * frame.shape[0])
                            cv2.line(frame, (xa, ya), (xb, yb),
                                     (58, 130, 248), 2)

                        landmarks = []
                        for lm in handLms:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        prediction = self._model.predict([landmarks])
                        gesture    = prediction[0]

                        # Run automation
                        self._run_automation(gesture)

            # Overlay gesture text on frame
            label_text = f"  {gesture}  "
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(frame,
                          (8, 10), (8 + tw + 6, 10 + th + 10),
                          (13, 27, 42), -1)
            g_color = {"palm": (227, 179, 65),
                       "index": (88, 166, 255),
                       "right slide": (63, 185, 80),
                       "left slide":  (63, 185, 80),
                       "screenshot":  (163, 113, 247)}.get(gesture,
                                                           (139, 148, 158))
            cv2.putText(frame, label_text,
                        (10, 10 + th + 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, g_color, 1,
                        cv2.LINE_AA)

            # Convert to PIL and queue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(rgb_frame)
            self.current_gesture = gesture

            try:
                self.frame_queue.put_nowait((pil_img, gesture))
            except queue.Full:
                pass

        cap.release()
        self._make_placeholder()
        self._set_status("Idle", TEXT_SEC)

    def _run_automation(self, gesture):
        if not self.automation_on.get():
            return
        current_t = time.time()
        cd        = self.cooldown_var.get()
        if current_t - self.last_action_t < cd:
            return

        if gesture == "palm":
            pyautogui.press('space')
            self._log("🖐", "Palm → Space (Play/Pause)", WARNING)
        elif gesture == "index":
            pyautogui.press('volumemute')
            self._log("☝",  "Index → Mute / Unmute", ACCENT)
        elif gesture == "right slide":
            pyautogui.scroll(-300)
            self._log("👉", "Right Slide → Scroll Up", SUCCESS)
        elif gesture == "left slide":
            pyautogui.scroll(300)
            self._log("👈", "Left Slide → Scroll Down", SUCCESS)
        elif gesture == "screenshot":
            fname = f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            pyautogui.screenshot(fname)
            self._log("📸", f"Screenshot → {fname}", "#a371f7")
        else:
            return  # No action for "None"

        self.last_action_t = current_t
        self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
        self.after(0, self._update_stats)

    
    #  UI Update helpers (called from background thread via after)
    
    def _poll_frame(self):
        """Pull frames from queue and update camera label — runs on main thread."""
        try:
            while True:
                pil_img, gesture = self.frame_queue.get_nowait()
                # Resize to fit label
                pil_img = pil_img.resize((620, 440), Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_img,
                                       dark_image=pil_img,
                                       size=(620, 440))
                self._cam_label.configure(image=ctk_img, text="")
                self._cam_label._image_ref = ctk_img
                self._update_gesture_badge(gesture)
        except queue.Empty:
            pass
        self.after(16, self._poll_frame)   # ~60 fps poll

    def _update_gesture_badge(self, gesture):
        icon, color = GESTURE_ICONS.get(gesture, ("✋", TEXT_SEC))
        self._gesture_icon_lbl.configure(text=icon)
        self._gesture_name_lbl.configure(text=gesture.title(),
                                         text_color=color)
        # Simulated confidence
        conf = 0.95 if gesture != "None" else 0.0
        self._conf_bar.set(conf)
        self._conf_bar.configure(progress_color=color)

    def _update_stats(self):
        total = max(sum(self.gesture_counts.values()), 1)
        for g, (bar, cnt_lbl) in self._stat_labels.items():
            c = self.gesture_counts.get(g, 0)
            bar.set(c / total)
            cnt_lbl.configure(text=str(c))

    def _clear_log(self):
        for w in list(self._log_panel._rows):
            w.destroy()
        self._log_panel._rows.clear()
        self._log_count_lbl.configure(text="0 events")

    def _log(self, icon, msg, color=TEXT_PRI):
        def _do():
            self._log_panel.add_entry(icon, msg, color)
            total = len(self._log_panel._rows)
            self._log_count_lbl.configure(text=f"{total} events")
        self.after(0, _do)

    def _set_status(self, text, color):
        def _do():
            self._status_label.configure(text=text, text_color=color)
            self._status_dot.set_color(color)
        self.after(0, _do)

    def _on_cooldown_change(self, val):
        self._cooldown_val_lbl.configure(text=f"{val:.1f}s")

    def _on_close(self):
        self.running = False
        self.after(200, self.destroy)



#  Entry point

if __name__ == "__main__":
    app = GestureUI()
    app.mainloop()
