/**
 * GESTURE AI — Real-Time Edge Vision & Gesture Automation Engine
 * Powered by MediaPipe Hands & Web Audio API
 */

// --- Global State & Configuration ---
const config = {
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
  cooldownMs: 1000,
  soundFxEnabled: true,
  soundFxVolume: 0.5,
  mirrorCam: true,
  drawLandmarks: true,
  isCameraRunning: false,
  isSimMode: false
};

const state = {
  lastActionTime: 0,
  currentGesture: "None",
  confidence: 0,
  handsDetected: 0,
  fingerStates: { thumb: false, index: false, middle: false, ring: false, pinky: false },
  fps: 0,
  frameCount: 0,
  lastFpsUpdateTime: performance.now(),
  motionHistory: [],
  snapshots: [],
  mediaPlaying: false,
  mediaMuted: false,
  mediaVolume: 0.8,
  activeFeedIndex: 0
};

// Joint Names Mapping (MediaPipe 21 Hand Landmarks)
const LANDMARK_NAMES = [
  "WRIST (0)", "THUMB_CMC (1)", "THUMB_MCP (2)", "THUMB_IP (3)", "THUMB_TIP (4)",
  "INDEX_MCP (5)", "INDEX_PIP (6)", "INDEX_DIP (7)", "INDEX_TIP (8)",
  "MIDDLE_MCP (9)", "MIDDLE_PIP (10)", "MIDDLE_DIP (11)", "MIDDLE_TIP (12)",
  "RING_MCP (13)", "RING_PIP (14)", "RING_DIP (15)", "RING_TIP (16)",
  "PINKY_MCP (17)", "PINKY_PIP (18)", "PINKY_DIP (19)", "PINKY_TIP (20)"
];

const GESTURE_ICONS = {
  "palm": "🖐",
  "index": "☝",
  "right slide": "👉",
  "left slide": "👈",
  "screenshot": "📸",
  "thumb_up": "👍",
  "peace": "✌",
  "fist": "✊",
  "None": "✋"
};

// --- DOM Element References ---
const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const canvasContainer = document.getElementById("canvasContainer");

const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const fpsCounter = document.getElementById("fpsCounter");
const latencyCounter = document.getElementById("latencyCounter");
const handCountBadge = document.getElementById("handCountBadge");

const hudGestureIcon = document.getElementById("hudGestureIcon");
const hudGestureName = document.getElementById("hudGestureName");
const hudConfidenceBar = document.getElementById("hudConfidenceBar");
const hudConfidenceVal = document.getElementById("hudConfidenceVal");
const cooldownBadge = document.getElementById("cooldownBadge");
const cooldownLabel = document.getElementById("cooldownLabel");
const actionAlert = document.getElementById("actionAlert");
const actionAlertText = document.getElementById("actionAlertText");
const cameraFallback = document.getElementById("cameraFallback");

// Finger Elements
const fingerElements = {
  thumb: document.getElementById("fingerThumb"),
  index: document.getElementById("fingerIndex"),
  middle: document.getElementById("fingerMiddle"),
  ring: document.getElementById("fingerRing"),
  pinky: document.getElementById("fingerPinky")
};

// --- Web Audio FX Synthesizer ---
let audioCtx = null;
function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function playSynthesizedSound(type) {
  if (!config.soundFxEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const masterGain = ctx.createGain();
    masterGain.gain.setValueAtTime(config.soundFxVolume * 0.25, now);
    masterGain.connect(ctx.destination);

    if (type === "play") {
      // Upward arpeggio
      [523.25, 659.25, 783.99].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, now + i * 0.08);
        gain.gain.setValueAtTime(0.3, now + i * 0.08);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.25);
        osc.connect(gain);
        gain.connect(masterGain);
        osc.start(now + i * 0.08);
        osc.stop(now + i * 0.08 + 0.25);
      });
    } else if (type === "pause") {
      // Downward chime
      [783.99, 523.25].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, now + i * 0.1);
        gain.gain.setValueAtTime(0.3, now + i * 0.1);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.1 + 0.3);
        osc.connect(gain);
        gain.connect(masterGain);
        osc.start(now + i * 0.1);
        osc.stop(now + i * 0.1 + 0.3);
      });
    } else if (type === "mute") {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(state.mediaMuted ? 350 : 600, now);
      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
      osc.connect(gain);
      gain.connect(masterGain);
      osc.start(now);
      osc.stop(now + 0.15);
    } else if (type === "scroll") {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, now);
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
      osc.connect(gain);
      gain.connect(masterGain);
      osc.start(now);
      osc.stop(now + 0.08);
    } else if (type === "screenshot") {
      // Camera Shutter white noise burst
      const bufferSize = ctx.sampleRate * 0.12;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = ctx.createBufferSource();
      noise.buffer = buffer;
      const noiseGain = ctx.createGain();
      noiseGain.gain.setValueAtTime(0.4, now);
      noiseGain.gain.exponentialRampToValueAtTime(0.01, now + 0.12);
      noise.connect(noiseGain);
      noiseGain.connect(masterGain);
      noise.start(now);
    } else if (type === "confetti") {
      // Major chord
      [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(freq, now + i * 0.05);
        gain.gain.setValueAtTime(0.25, now + i * 0.05);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.05 + 0.4);
        osc.connect(gain);
        gain.connect(masterGain);
        osc.start(now + i * 0.05);
        osc.stop(now + i * 0.05 + 0.4);
      });
    }
  } catch (e) {
    console.warn("Audio FX error:", e);
  }
}

// --- Gesture Classification Engine ---
function classifyHandGesture(landmarks) {
  if (!landmarks || landmarks.length < 21) {
    return { name: "None", confidence: 0, fingerStates: {} };
  }

  // Joint positions
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];

  const indexTip = landmarks[8];
  const indexDip = landmarks[7];
  const indexPip = landmarks[6];
  const indexMcp = landmarks[5];

  const middleTip = landmarks[12];
  const middlePip = landmarks[10];

  const ringTip = landmarks[16];
  const ringPip = landmarks[14];

  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];

  // Helper Euclidean distance
  const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

  // Finger Extension status
  // For index, middle, ring, pinky: Tip is higher (smaller Y) than PIP
  const indexExtended = indexTip.y < indexPip.y;
  const middleExtended = middleTip.y < middlePip.y;
  const ringExtended = ringTip.y < ringPip.y;
  const pinkyExtended = pinkyTip.y < pinkyPip.y;

  // Thumb Extension: distance from thumb tip to pinky MCP compared to thumb MCP
  const thumbDistToPinky = dist(thumbTip, pinkyPip);
  const thumbMcpDistToPinky = dist(thumbMcp, pinkyPip);
  const thumbExtended = thumbDistToPinky > thumbMcpDistToPinky * 1.15;

  const fingerStates = {
    thumb: thumbExtended,
    index: indexExtended,
    middle: middleExtended,
    ring: ringExtended,
    pinky: pinkyExtended
  };

  // Motion analysis (track wrist / index X velocity)
  const now = performance.now();
  state.motionHistory.push({ time: now, x: wrist.x, y: wrist.y, indexX: indexTip.x });
  if (state.motionHistory.length > 10) state.motionHistory.shift();

  let deltaX = 0;
  if (state.motionHistory.length >= 4) {
    const oldest = state.motionHistory[0];
    const newest = state.motionHistory[state.motionHistory.length - 1];
    deltaX = newest.x - oldest.x;
  }

  // 1. Screenshot / Pinch Gesture: Thumb Tip and Index Tip touching or very close
  const pinchDist = dist(thumbTip, indexTip);
  if (pinchDist < 0.075 && !middleExtended && !ringExtended && !pinkyExtended) {
    return { name: "screenshot", confidence: 0.96, fingerStates };
  }

  // 2. Thumbs Up Gesture: Thumb extended upward (tip.y < ip.y), all other 4 fingers folded
  if (thumbTip.y < thumbIp.y - 0.04 && !indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
    return { name: "thumb_up", confidence: 0.95, fingerStates };
  }

  // 3. Palm Gesture: All 5 fingers extended
  if (thumbExtended && indexExtended && middleExtended && ringExtended && pinkyExtended) {
    return { name: "palm", confidence: 0.98, fingerStates };
  }

  // 4. Index Point Gesture: Index extended, others folded
  if (indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
    // Check horizontal pointing direction for Slide gestures
    const isPointingRight = indexTip.x > indexMcp.x + 0.08 && Math.abs(indexTip.y - indexMcp.y) < 0.12;
    const isPointingLeft = indexTip.x < indexMcp.x - 0.08 && Math.abs(indexTip.y - indexMcp.y) < 0.12;

    if (isPointingRight || deltaX > 0.08) {
      return { name: "right slide", confidence: 0.92, fingerStates };
    } else if (isPointingLeft || deltaX < -0.08) {
      return { name: "left slide", confidence: 0.92, fingerStates };
    }

    return { name: "index", confidence: 0.96, fingerStates };
  }

  // 5. Left / Right Slide motion gestures (hand moving with open palm or partial fingers)
  if (deltaX > 0.09) {
    return { name: "right slide", confidence: 0.88, fingerStates };
  } else if (deltaX < -0.09) {
    return { name: "left slide", confidence: 0.88, fingerStates };
  }

  // 6. Peace / Victory Sign
  if (indexExtended && middleExtended && !ringExtended && !pinkyExtended) {
    return { name: "peace", confidence: 0.94, fingerStates };
  }

  // 7. Fist Gesture
  if (!thumbExtended && !indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
    return { name: "fist", confidence: 0.93, fingerStates };
  }

  return { name: "None", confidence: 0.5, fingerStates };
}

// --- Action Execution & Cooldown Dispatcher ---
function handleGestureAction(gestureName, confidence) {
  if (!gestureName || gestureName === "None") {
    updateActiveGesturePill(null);
    return;
  }

  updateActiveGesturePill(gestureName);

  const now = performance.now();
  const timeSinceLast = now - state.lastActionTime;

  // Check cooldown
  if (timeSinceLast < config.cooldownMs) {
    updateCooldownVisual(timeSinceLast, config.cooldownMs);
    return;
  }

  // Execute Action
  state.lastActionTime = now;
  updateCooldownVisual(0, config.cooldownMs);

  switch (gestureName) {
    case "palm":
      togglePlayPause();
      triggerAlert("🖐 Palm Detected &bull; Toggled Play/Pause");
      playSynthesizedSound(state.mediaPlaying ? "play" : "pause");
      logEvent(`Gesture Action: [Palm] -> ${state.mediaPlaying ? "Playing" : "Paused"} Media`, "action");
      break;

    case "index":
      toggleMute();
      triggerAlert(`☝ Index Detected &bull; ${state.mediaMuted ? "Muted" : "Unmuted"} Audio`);
      playSynthesizedSound("mute");
      logEvent(`Gesture Action: [Index] -> Toggled Mute (${state.mediaMuted ? "Muted" : "Active"})`, "action");
      break;

    case "right slide":
      scrollFeed(1);
      triggerAlert("👉 Right Slide &bull; Scrolled Feed Down");
      playSynthesizedSound("scroll");
      logEvent("Gesture Action: [Right Slide] -> Scrolled Feed Down", "action");
      break;

    case "left slide":
      scrollFeed(-1);
      triggerAlert("👈 Left Slide &bull; Scrolled Feed Up");
      playSynthesizedSound("scroll");
      logEvent("Gesture Action: [Left Slide] -> Scrolled Feed Up", "action");
      break;

    case "screenshot":
      captureLiveSnapshot();
      triggerAlert("📸 Screenshot Gesture &bull; Snapshot Saved!");
      playSynthesizedSound("screenshot");
      logEvent("Gesture Action: [Screenshot] -> Captured Snapshot Frame", "action");
      break;

    case "thumb_up":
      triggerConfetti();
      triggerAlert("👍 Thumbs Up &bull; Awesome Job!");
      playSynthesizedSound("confetti");
      logEvent("Gesture Action: [Thumbs Up] -> Triggered Celebration Confetti", "action");
      break;

    default:
      break;
  }
}

// --- Interactive Studio Functions ---
function togglePlayPause() {
  state.mediaPlaying = !state.mediaPlaying;
  const playBtnIcon = document.getElementById("playPauseIcon");
  const playerStatus = document.getElementById("playerStatusBadge");

  if (state.mediaPlaying) {
    playBtnIcon.className = "fa-solid fa-pause";
    playerStatus.textContent = "Playing";
    playerStatus.className = "badge badge-accent";
  } else {
    playBtnIcon.className = "fa-solid fa-play";
    playerStatus.textContent = "Paused";
    playerStatus.className = "badge";
  }
}

function toggleMute() {
  state.mediaMuted = !state.mediaMuted;
  const muteBtnIcon = document.getElementById("muteBtnIcon");
  if (state.mediaMuted) {
    muteBtnIcon.className = "fa-solid fa-volume-xmark";
    muteBtnIcon.style.color = "var(--accent-rose)";
  } else {
    muteBtnIcon.className = "fa-solid fa-volume-high";
    muteBtnIcon.style.color = "var(--text-primary)";
  }
}

function scrollFeed(direction) {
  const container = document.getElementById("feedScrollContainer");
  const items = container.querySelectorAll(".feed-item");
  state.activeFeedIndex = Math.max(0, Math.min(items.length - 1, state.activeFeedIndex + direction));

  items.forEach((item, idx) => {
    item.classList.toggle("active-feed-item", idx === state.activeFeedIndex);
  });

  const targetItem = items[state.activeFeedIndex];
  if (targetItem) {
    targetItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function captureLiveSnapshot() {
  if (!canvasElement) return;

  // Flash effect on canvas container
  canvasContainer.style.boxShadow = "0 0 50px rgba(255, 255, 255, 0.9)";
  setTimeout(() => {
    canvasContainer.style.boxShadow = "";
  }, 200);

  const dataUrl = canvasElement.toDataURL("image/png");
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0];

  const snapshotObj = { dataUrl, time: timeStr, id: Date.now() };
  state.snapshots.unshift(snapshotObj);
  if (state.snapshots.length > 8) state.snapshots.pop();

  renderSnapshotReel();
}

function renderSnapshotReel() {
  const reel = document.getElementById("snapshotReel");
  const emptyState = document.getElementById("snapshotEmptyState");
  const countBadge = document.getElementById("snapshotCountBadge");

  countBadge.textContent = `${state.snapshots.length} Captures`;

  if (state.snapshots.length === 0) {
    emptyState.style.display = "flex";
    reel.innerHTML = "";
    reel.appendChild(emptyState);
    return;
  }

  emptyState.style.display = "none";
  reel.innerHTML = "";

  state.snapshots.forEach((snap) => {
    const thumb = document.createElement("div");
    thumb.className = "snapshot-thumb";
    thumb.innerHTML = `
      <img src="${snap.dataUrl}" alt="Snapshot at ${snap.time}" />
      <span class="snapshot-time-tag">${snap.time}</span>
    `;
    thumb.addEventListener("click", () => openSnapshotModal(snap.dataUrl));
    reel.appendChild(thumb);
  });
}

function openSnapshotModal(dataUrl) {
  const modal = document.getElementById("snapshotModal");
  const img = document.getElementById("modalSnapshotImg");
  const downloadLink = document.getElementById("downloadSnapshotLink");

  img.src = dataUrl;
  downloadLink.href = dataUrl;
  downloadLink.download = `gesture_snapshot_${Date.now()}.png`;

  modal.classList.add("open");
}

function triggerAlert(msg) {
  actionAlertText.innerHTML = msg;
  actionAlert.classList.add("show");
  clearTimeout(actionAlert._timer);
  actionAlert._timer = setTimeout(() => {
    actionAlert.classList.remove("show");
  }, 2000);
}

function updateCooldownVisual(elapsed, total) {
  if (elapsed >= total || elapsed === 0) {
    cooldownBadge.classList.remove("cooling");
    cooldownLabel.textContent = "READY";
  } else {
    cooldownBadge.classList.add("cooling");
    const remaining = ((total - elapsed) / 1000).toFixed(1);
    cooldownLabel.textContent = `${remaining}s`;
  }
}

function updateActiveGesturePill(gestureName) {
  document.querySelectorAll(".gesture-pill").forEach((pill) => {
    const pillGesture = pill.getAttribute("data-gesture");
    pill.classList.toggle("active-gesture", pillGesture === gestureName);
  });
}

function updateFingerMatrix(fingerStates) {
  if (!fingerStates) return;
  for (const [finger, extended] of Object.entries(fingerStates)) {
    const el = fingerElements[finger];
    if (el) {
      el.classList.toggle("extended", !!extended);
      el.querySelector(".finger-status").textContent = extended ? "UP" : "FOLDED";
    }
  }
}

function updateLandmarkTable(landmarks) {
  const tbody = document.getElementById("landmarkTableBody");
  if (!landmarks || landmarks.length < 21) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-table-msg">No hand detected in current frame</td></tr>`;
    return;
  }

  let html = "";
  for (let i = 0; i < 21; i++) {
    const lm = landmarks[i];
    html += `
      <tr>
        <td>${i}</td>
        <td>${LANDMARK_NAMES[i]}</td>
        <td>${lm.x.toFixed(3)}</td>
        <td>${lm.y.toFixed(3)}</td>
        <td>${lm.z.toFixed(3)}</td>
      </tr>
    `;
  }
  tbody.innerHTML = html;
}

function logEvent(msg, type = "info") {
  const consoleEl = document.getElementById("logConsole");
  const time = new Date().toTimeString().split(" ")[0];
  const entry = document.createElement("div");
  entry.className = `log-entry log-${type}`;
  entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-msg">${msg}</span>`;
  consoleEl.insertBefore(entry, consoleEl.firstChild);
  if (consoleEl.children.length > 50) consoleEl.removeChild(consoleEl.lastChild);
}

// --- Audio Visualizer Waveform Generator ---
const visualizerCanvas = document.getElementById("audioVisualizerCanvas");
const visualizerCtx = visualizerCanvas ? visualizerCanvas.getContext("2d") : null;
let visualizerPhase = 0;

function drawAudioVisualizer() {
  if (!visualizerCtx) return;
  const w = visualizerCanvas.width;
  const h = visualizerCanvas.height;

  visualizerCtx.clearRect(0, 0, w, h);

  const bars = 32;
  const barWidth = w / bars - 2;

  for (let i = 0; i < bars; i++) {
    let barHeight = 4;
    if (state.mediaPlaying && !state.mediaMuted) {
      const freq = (i + 1) * 0.3 + visualizerPhase;
      barHeight = (Math.sin(freq) * 0.5 + 0.5) * (h - 15) * state.mediaVolume + 6;
    }

    const x = i * (barWidth + 2);
    const y = h - barHeight;

    const grad = visualizerCtx.createLinearGradient(0, y, 0, h);
    grad.addColorStop(0, "#00f2fe");
    grad.addColorStop(1, "#4facfe");

    visualizerCtx.fillStyle = grad;
    visualizerCtx.fillRect(x, y, barWidth, barHeight);
  }

  if (state.mediaPlaying) {
    visualizerPhase += 0.15;
    const progressFill = document.getElementById("trackProgress");
    if (progressFill) {
      const curWidth = parseFloat(progressFill.style.width || "25");
      progressFill.style.width = `${(curWidth + 0.05) % 100}%`;
    }
  }

  requestAnimationFrame(drawAudioVisualizer);
}
requestAnimationFrame(drawAudioVisualizer);

// --- Confetti Particle System ---
const confettiCanvas = document.getElementById("confettiCanvas");
const confettiCtx = confettiCanvas ? confettiCanvas.getContext("2d") : null;
let confettiParticles = [];

function triggerConfetti() {
  if (!confettiCanvas) return;
  confettiCanvas.width = window.innerWidth;
  confettiCanvas.height = window.innerHeight;

  const colors = ["#00f2fe", "#4facfe", "#00f5a0", "#ff2a85", "#ffb703", "#9d4edd"];
  for (let i = 0; i < 80; i++) {
    confettiParticles.push({
      x: window.innerWidth / 2 + (Math.random() * 200 - 100),
      y: window.innerHeight / 2 + (Math.random() * 100 - 50),
      vx: (Math.random() - 0.5) * 12,
      vy: Math.random() * -12 - 4,
      size: Math.random() * 8 + 4,
      color: colors[Math.floor(Math.random() * colors.length)],
      rotation: Math.random() * 360,
      vRot: (Math.random() - 0.5) * 10,
      alpha: 1
    });
  }
}

function updateConfetti() {
  if (!confettiCtx || confettiParticles.length === 0) {
    requestAnimationFrame(updateConfetti);
    return;
  }
  confettiCtx.clearRect(0, 0, confettiCanvas.width, confettiCanvas.height);

  for (let i = confettiParticles.length - 1; i >= 0; i--) {
    const p = confettiParticles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.35; // gravity
    p.rotation += p.vRot;
    p.alpha -= 0.012;

    if (p.alpha <= 0 || p.y > window.innerHeight) {
      confettiParticles.splice(i, 1);
      continue;
    }

    confettiCtx.save();
    confettiCtx.translate(p.x, p.y);
    confettiCtx.rotate((p.rotation * Math.PI) / 180);
    confettiCtx.fillStyle = p.color;
    confettiCtx.globalAlpha = p.alpha;
    confettiCtx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
    confettiCtx.restore();
  }

  requestAnimationFrame(updateConfetti);
}
requestAnimationFrame(updateConfetti);

// --- MediaPipe Results Callback ---
function onHandsResults(results) {
  const startTime = performance.now();

  // Resize canvas match container
  if (canvasElement.width !== canvasElement.clientWidth || canvasElement.height !== canvasElement.clientHeight) {
    canvasElement.width = canvasElement.clientWidth || 640;
    canvasElement.height = canvasElement.clientHeight || 480;
  }

  const w = canvasElement.width;
  const h = canvasElement.height;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, w, h);

  if (config.mirrorCam) {
    canvasCtx.translate(w, 0);
    canvasCtx.scale(-1, 1);
  }

  // Draw Camera Frame
  if (results.image) {
    canvasCtx.drawImage(results.image, 0, 0, w, h);
  }

  let detectedGesture = "None";
  let detectedConfidence = 0;
  let activeFingerStates = { thumb: false, index: false, middle: false, ring: false, pinky: false };
  let primaryLandmarks = null;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    state.handsDetected = results.multiHandLandmarks.length;
    handCountBadge.textContent = `${state.handsDetected} Hand${state.handsDetected > 1 ? "s" : ""} Active`;
    handCountBadge.className = "badge badge-accent";

    primaryLandmarks = results.multiHandLandmarks[0];

    // Draw Skeleton Landmarks
    if (config.drawLandmarks && window.drawConnectors && window.drawLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {
          color: "#00f2fe",
          lineWidth: 3
        });
        window.drawLandmarks(canvasCtx, landmarks, {
          color: "#ff2a85",
          fillColor: "#00f5a0",
          lineWidth: 1.5,
          radius: 4
        });
      }
    }

    // Classify Gesture
    const classification = classifyHandGesture(primaryLandmarks);
    detectedGesture = classification.name;
    detectedConfidence = classification.confidence;
    activeFingerStates = classification.fingerStates;
  } else {
    state.handsDetected = 0;
    handCountBadge.textContent = "0 Hands Detected";
    handCountBadge.className = "badge";
  }

  canvasCtx.restore();

  // Update State & UI
  state.currentGesture = detectedGesture;
  state.confidence = detectedConfidence;
  state.fingerStates = activeFingerStates;

  // Update HUD
  hudGestureIcon.textContent = GESTURE_ICONS[detectedGesture] || "✋";
  hudGestureName.textContent = detectedGesture;
  const pct = Math.round(detectedConfidence * 100);
  hudConfidenceBar.style.width = `${pct}%`;
  hudConfidenceVal.textContent = `${pct}%`;

  updateFingerMatrix(activeFingerStates);
  updateLandmarkTable(primaryLandmarks);

  // Dispatch Gesture Actions
  if (detectedConfidence >= config.minDetectionConfidence) {
    handleGestureAction(detectedGesture, detectedConfidence);
  }

  // FPS & Latency calculation
  state.frameCount++;
  const now = performance.now();
  if (now - state.lastFpsUpdateTime >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFpsUpdateTime));
    fpsCounter.textContent = `${state.fps} FPS`;
    state.frameCount = 0;
    state.lastFpsUpdateTime = now;
  }
  const latency = Math.round(performance.now() - startTime);
  latencyCounter.textContent = `${latency} ms`;
}

// --- MediaPipe Hands Initialization ---
let handsInstance = null;
let cameraInstance = null;

function initMediaPipeHands() {
  statusText.textContent = "Loading AI Vision Model...";
  statusPill.querySelector(".status-dot").style.backgroundColor = "var(--accent-amber)";

  if (typeof Hands === "undefined") {
    console.error("MediaPipe Hands not loaded from CDN.");
    statusText.textContent = "CDN Offline &bull; Switch to Sim Mode";
    statusPill.querySelector(".status-dot").style.backgroundColor = "var(--accent-rose)";
    return;
  }

  handsInstance = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  handsInstance.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: config.minDetectionConfidence,
    minTrackingConfidence: config.minTrackingConfidence
  });

  handsInstance.onResults(onHandsResults);

  statusText.textContent = "AI Vision Engine Ready";
  statusPill.querySelector(".status-dot").style.backgroundColor = "var(--accent-emerald)";
  logEvent("MediaPipe Hands 2.0 neural vision engine initialized successfully.", "info");
}

async function startCamera() {
  if (!videoElement) return;

  try {
    statusText.textContent = "Starting WebCam...";

    if (!cameraInstance && typeof Camera !== "undefined") {
      cameraInstance = new Camera(videoElement, {
        onFrame: async () => {
          if (config.isCameraRunning && handsInstance) {
            await handsInstance.send({ image: videoElement });
          }
        },
        width: 640,
        height: 480
      });
    }

    if (cameraInstance) {
      await cameraInstance.start();
      config.isCameraRunning = true;
      cameraFallback.classList.add("hidden");
      statusText.textContent = "Live WebCam Streaming";
      statusPill.querySelector(".status-dot").style.backgroundColor = "var(--accent-emerald)";
      logEvent("Camera stream connected (640x480). Real-time tracking online.", "info");
    } else {
      throw new Error("Camera utility not available");
    }
  } catch (err) {
    console.warn("Camera start failed, prompting simulation mode:", err);
    statusText.textContent = "Camera Blocked / Unavailable";
    statusPill.querySelector(".status-dot").style.backgroundColor = "var(--accent-rose)";
    cameraFallback.classList.remove("hidden");
    logEvent(`WebCam access error: ${err.message}. Try Preset Simulation mode.`, "warn");
  }
}

function stopCamera() {
  config.isCameraRunning = false;
  cameraFallback.classList.remove("hidden");
  statusText.textContent = "WebCam Stopped";
  statusPill.querySelector(".status-dot").style.backgroundColor = "var(--accent-amber)";
  logEvent("Camera stream stopped by user.", "info");
}

// --- Simulation / Preset Mode Handler ---
function triggerPresetGesture(gestureName) {
  logEvent(`Simulation: Manually triggered preset gesture [${gestureName}]`, "info");
  handleGestureAction(gestureName, 1.0);

  hudGestureIcon.textContent = GESTURE_ICONS[gestureName] || "✋";
  hudGestureName.textContent = gestureName;
  hudConfidenceBar.style.width = "100%";
  hudConfidenceVal.textContent = "100% (Sim)";

  // Draw simulated visualization on canvas
  drawSimulatedGesture(gestureName);
}

function drawSimulatedGesture(gestureName) {
  const w = canvasElement.width || 640;
  const h = canvasElement.height || 480;
  canvasCtx.clearRect(0, 0, w, h);

  // Background Cyber Grid
  canvasCtx.fillStyle = "#06090e";
  canvasCtx.fillRect(0, 0, w, h);

  canvasCtx.strokeStyle = "rgba(0, 242, 254, 0.1)";
  canvasCtx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    canvasCtx.beginPath();
    canvasCtx.moveTo(x, 0);
    canvasCtx.lineTo(x, h);
    canvasCtx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, y);
    canvasCtx.lineTo(w, y);
    canvasCtx.stroke();
  }

  // Draw Center Hologram
  canvasCtx.font = "80px Outfit, sans-serif";
  canvasCtx.textAlign = "center";
  canvasCtx.textBaseline = "middle";
  canvasCtx.fillText(GESTURE_ICONS[gestureName] || "🖐", w / 2, h / 2 - 20);

  canvasCtx.font = "bold 24px 'JetBrains Mono', monospace";
  canvasCtx.fillStyle = "#00f2fe";
  canvasCtx.fillText(`[PRESET: ${gestureName.toUpperCase()}]`, w / 2, h / 2 + 60);

  canvasCtx.font = "14px Outfit, sans-serif";
  canvasCtx.fillStyle = "#94a3b8";
  canvasCtx.fillText("Action Dispatched Successfully", w / 2, h / 2 + 95);
}

// --- UI Event Listeners & Binding ---
function setupEventListeners() {
  // Start Camera Buttons
  const startCameraBtn = document.getElementById("startCameraBtn");
  if (startCameraBtn) startCameraBtn.addEventListener("click", () => startCamera());

  const startSimPresetBtn = document.getElementById("startSimPresetBtn");
  if (startSimPresetBtn) {
    startSimPresetBtn.addEventListener("click", () => {
      cameraFallback.classList.add("hidden");
      triggerPresetGesture("palm");
    });
  }

  // Toolbar Buttons
  const toggleCamBtn = document.getElementById("toggleCamBtn");
  if (toggleCamBtn) {
    toggleCamBtn.addEventListener("click", () => {
      if (config.isCameraRunning) {
        stopCamera();
        toggleCamBtn.classList.remove("active");
      } else {
        startCamera();
        toggleCamBtn.classList.add("active");
      }
    });
  }

  const mirrorCamBtn = document.getElementById("mirrorCamBtn");
  if (mirrorCamBtn) {
    mirrorCamBtn.addEventListener("click", () => {
      config.mirrorCam = !config.mirrorCam;
      mirrorCamBtn.classList.toggle("active", config.mirrorCam);
    });
  }

  const drawLandmarksBtn = document.getElementById("drawLandmarksBtn");
  if (drawLandmarksBtn) {
    drawLandmarksBtn.addEventListener("click", () => {
      config.drawLandmarks = !config.drawLandmarks;
      drawLandmarksBtn.classList.toggle("active", config.drawLandmarks);
    });
  }

  const captureSnapshotBtn = document.getElementById("captureSnapshotBtn");
  if (captureSnapshotBtn) {
    captureSnapshotBtn.addEventListener("click", () => {
      captureLiveSnapshot();
      playSynthesizedSound("screenshot");
      triggerAlert("📸 Snapshot Captured!");
    });
  }

  const resetTrackerBtn = document.getElementById("resetTrackerBtn");
  if (resetTrackerBtn) {
    resetTrackerBtn.addEventListener("click", () => {
      state.motionHistory = [];
      logEvent("Neural tracker reset.", "info");
      triggerAlert("🔄 Tracker Reset");
    });
  }

  // Sound FX Toggle
  const soundToggleBtn = document.getElementById("soundToggleBtn");
  const soundIcon = document.getElementById("soundIcon");
  if (soundToggleBtn) {
    soundToggleBtn.addEventListener("click", () => {
      config.soundFxEnabled = !config.soundFxEnabled;
      if (config.soundFxEnabled) {
        soundIcon.className = "fa-solid fa-volume-high";
        triggerAlert("🔊 Sound FX Enabled");
      } else {
        soundIcon.className = "fa-solid fa-volume-xmark";
        triggerAlert("🔇 Sound FX Muted");
      }
    });
  }

  // Sim Mode Quick Trigger
  const simModeToggleBtn = document.getElementById("simModeToggleBtn");
  if (simModeToggleBtn) {
    simModeToggleBtn.addEventListener("click", () => {
      const presets = ["palm", "index", "right slide", "left slide", "screenshot", "thumb_up"];
      const rand = presets[Math.floor(Math.random() * presets.length)];
      triggerPresetGesture(rand);
    });
  }

  // Preset Buttons in Settings Modal
  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = btn.getAttribute("data-sim");
      triggerPresetGesture(preset);
      document.getElementById("settingsModal").classList.remove("open");
    });
  });

  // Settings Modal Open/Close
  const openSettingsBtn = document.getElementById("openSettingsBtn");
  const settingsModal = document.getElementById("settingsModal");
  const closeSettingsBtn = document.getElementById("closeSettingsBtn");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");

  if (openSettingsBtn) openSettingsBtn.addEventListener("click", () => settingsModal.classList.add("open"));
  if (closeSettingsBtn) closeSettingsBtn.addEventListener("click", () => settingsModal.classList.remove("open"));
  if (saveSettingsBtn) saveSettingsBtn.addEventListener("click", () => settingsModal.classList.remove("open"));

  // Settings Controls
  const confidenceSlider = document.getElementById("confidenceThreshold");
  const confidenceLabel = document.getElementById("confidenceValLabel");
  if (confidenceSlider) {
    confidenceSlider.addEventListener("input", (e) => {
      config.minDetectionConfidence = e.target.value / 100;
      confidenceLabel.textContent = `${e.target.value}%`;
      if (handsInstance) {
        handsInstance.setOptions({ minDetectionConfidence: config.minDetectionConfidence });
      }
    });
  }

  const cooldownSlider = document.getElementById("cooldownDuration");
  const cooldownValLabel = document.getElementById("cooldownValLabel");
  if (cooldownSlider) {
    cooldownSlider.addEventListener("input", (e) => {
      config.cooldownMs = e.target.value * 100;
      cooldownValLabel.textContent = `${(config.cooldownMs / 1000).toFixed(1)}s`;
    });
  }

  const soundVolumeSlider = document.getElementById("soundVolume");
  const soundVolumeValLabel = document.getElementById("soundVolumeValLabel");
  if (soundVolumeSlider) {
    soundVolumeSlider.addEventListener("input", (e) => {
      config.soundFxVolume = e.target.value / 100;
      soundVolumeValLabel.textContent = `${e.target.value}%`;
    });
  }

  // Snapshot Modal Close
  const snapshotModal = document.getElementById("snapshotModal");
  const closeSnapshotBtn = document.getElementById("closeSnapshotBtn");
  if (closeSnapshotBtn) closeSnapshotBtn.addEventListener("click", () => snapshotModal.classList.remove("open"));

  // Tab Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const targetId = `tab-${btn.getAttribute("data-tab")}`;
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  // Footer Docs Link
  const footerDocsLink = document.getElementById("footerDocsLink");
  if (footerDocsLink) {
    footerDocsLink.addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelector('[data-tab="desktop"]').click();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // Media Player Manual Buttons
  const playPauseBtn = document.getElementById("playPauseBtn");
  if (playPauseBtn) playPauseBtn.addEventListener("click", togglePlayPause);

  const muteBtn = document.getElementById("muteBtn");
  if (muteBtn) muteBtn.addEventListener("click", toggleMute);

  const prevTrackBtn = document.getElementById("prevTrackBtn");
  if (prevTrackBtn) prevTrackBtn.addEventListener("click", () => triggerAlert("⏮ Previous Track"));

  const nextTrackBtn = document.getElementById("nextTrackBtn");
  if (nextTrackBtn) nextTrackBtn.addEventListener("click", () => triggerAlert("⏭ Next Track"));

  const volumeSlider = document.getElementById("volumeSlider");
  if (volumeSlider) {
    volumeSlider.addEventListener("input", (e) => {
      state.mediaVolume = e.target.value / 100;
    });
  }

  // Clear Logs Button
  const clearLogsBtn = document.getElementById("clearLogsBtn");
  if (clearLogsBtn) {
    clearLogsBtn.addEventListener("click", () => {
      document.getElementById("logConsole").innerHTML = "";
    });
  }

  // Copy Code Snippets
  document.querySelectorAll(".copy-code-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-clipboard");
      if (text) {
        navigator.clipboard.writeText(text.replace(/`n/g, "\n")).then(() => {
          btn.innerHTML = `<i class="fa-solid fa-check" style="color:var(--accent-emerald)"></i>`;
          setTimeout(() => {
            btn.innerHTML = `<i class="fa-regular fa-copy"></i>`;
          }, 1500);
        });
      }
    });
  });

  // Click gesture pills to trigger preset
  document.querySelectorAll(".gesture-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const g = pill.getAttribute("data-gesture");
      triggerPresetGesture(g);
    });
  });
}

// --- Initialize App on Page Load ---
window.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  initMediaPipeHands();

  // Prompt user for camera
  startCamera();
});
