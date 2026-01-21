import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import pickle
import json
from collections import deque

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="StarkTouch Engine", layout="wide")
st.title("🖐️ StarkTouch Engine — Phase 1: Gesture-Controlled Virtual Canvas")

# Parameters
CANVAS_SIZE = (720, 480)
BRUSH_SIZE = 5
ERASE_SIZE = 30
FPS_TARGET = 60

# ==============================
# HAND TRACKING PIPELINE
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# EMA smoothing
def smooth_point(prev, new, alpha=0.2):
    return (1 - alpha) * np.array(prev) + alpha * np.array(new)

# ==============================
# STATE MACHINE
# ==============================
STATE_IDLE = "IDLE"
STATE_DRAW = "DRAW"
STATE_ERASE = "ERASE"
STATE_MOVE = "MOVE"
STATE_HOVER = "HOVER"

state = STATE_IDLE
debounce_counter = 0
DEBOUNCE_LIMIT = 5

# ==============================
# CANVAS + HISTORY
# ==============================
canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
stroke_history = deque(maxlen=500)

# ==============================
# STREAMLIT UI
# ==============================
hud = st.sidebar.checkbox("Show HUD", value=True)
save_btn = st.sidebar.button("💾 Save Drawing")
load_btn = st.sidebar.button("📂 Load Drawing")

# ==============================
# PERSISTENCE
# ==============================
def save_drawing():
    with open("drawing.pkl", "wb") as f:
        pickle.dump(stroke_history, f)
    st.sidebar.success("Drawing saved!")

def load_drawing():
    global stroke_history, canvas
    try:
        with open("drawing.pkl", "rb") as f:
            stroke_history = pickle.load(f)
        canvas = np.zeros_like(canvas)
        for stroke in stroke_history:
            cv2.circle(canvas, stroke, BRUSH_SIZE, (0, 255, 255), -1)
        st.sidebar.success("Drawing loaded!")
    except:
        st.sidebar.error("No saved drawing found.")

if save_btn:
    save_drawing()
if load_btn:
    load_drawing()

# ==============================
# MAIN LOOP
# ==============================
cap = cv2.VideoCapture(0)
prev_point = (0, 0)

frame_placeholder = st.empty()
fps_placeholder = st.sidebar.empty()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.error("Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract index fingertip
        x = int(hand_landmarks.landmark[8].x * CANVAS_SIZE[0])
        y = int(hand_landmarks.landmark[8].y * CANVAS_SIZE[1])
        point = smooth_point(prev_point, (x, y))
        prev_point = point.astype(int)

        # Gesture detection (simple distances)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        dist = np.linalg.norm(
            np.array([thumb_tip.x, thumb_tip.y]) -
            np.array([index_tip.x, index_tip.y])
        )

        # State machine
        if dist < 0.05:
            new_state = STATE_DRAW
        elif dist > 0.15:
            new_state = STATE_ERASE
        else:
            new_state = STATE_HOVER

        if new_state != state:
            debounce_counter += 1
            if debounce_counter > DEBOUNCE_LIMIT:
                state = new_state
                debounce_counter = 0
        else:
            debounce_counter = 0

        # Apply actions
        if state == STATE_DRAW:
            cv2.circle(canvas, tuple(prev_point), BRUSH_SIZE, (0, 255, 255), -1)
            stroke_history.append(tuple(prev_point))
        elif state == STATE_ERASE:
            cv2.circle(canvas, tuple(prev_point), ERASE_SIZE, (0, 0, 0), -1)

        # Draw neon glow
        glow = cv2.GaussianBlur(canvas, (0, 0), 5)
        frame = cv2.addWeighted(frame, 1.0, glow, 0.6, 0)

        # HUD
        if hud:
            cv2.putText(frame, f"Mode: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_placeholder.text(f"FPS: {fps:.1f}")

    # Display
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
