import cv2
import mediapipe as mp
import math
import time
import numpy as np
from collections import deque
import requests

# ====== CONFIG ======
ESP32_IP = "192.168.78.93"  # เปลี่ยนเป็น IP ของคุณ
FLASH_ON_URL = f"http://{ESP32_IP}/flash/on"
FLASH_OFF_URL = f"http://{ESP32_IP}/flash/off"
CAM_STREAM_URL = f"http://{ESP32_IP}/stream"

MAX_DISTANCE = 0.35
MIN_DISTANCE = 0.02

# ====== MediaPipe setup ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ====== โหมด ======
MODES = ["NORMAL", "SYMBOL", "VOLUME_CONTROL", "FLASH_CONTROL"]
current_mode = "NORMAL"

# ====== เวลาและสถานะ ======
OPEN_HAND_DURATION = 2.0
OK_HOLD_DURATION = 1.0
GRACE_PERIOD = 0.4
STABILITY_THRESHOLD = 0.008

mode_transition_start_time = None
mode_transition_grace_time = None
ok_hold_start_time = None

# ====== สถานะมือ ======
class HandData:
    def __init__(self):
        self.finger_count = 0
        self.volume_level = 50
        self.distance_history = deque(maxlen=30)
        self.is_detected = False
        self.landmarks = None

left_hand_data = HandData()
right_hand_data = HandData()

# ====== แฟลชสถานะ ======
flash_on = False

# ====== ฟังก์ชันเปิด/ปิดแฟลช ======
def flash_on_request():
    try:
        r = requests.get(FLASH_ON_URL, timeout=2)
        print("Flash ON response:", r.text)
    except Exception as e:
        print("Error sending flash ON request:", e)

def flash_off_request():
    try:
        r = requests.get(FLASH_OFF_URL, timeout=2)
        print("Flash OFF response:", r.text)
    except Exception as e:
        print("Error sending flash OFF request:", e)

# ====== ฟังก์ชันช่วย ======

def get_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

def is_ok_symbol(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = get_distance(thumb_tip, index_tip)
    touching = dist < 0.06

    def finger_up(tip, pip):
        return landmarks[tip].y < landmarks[pip].y

    return (touching and
            finger_up(12, 10) and
            finger_up(16, 14) and
            finger_up(20, 18))

def count_fingers_strict(landmarks, hand_label):
    if hand_label == "Right":
        thumb_open = landmarks[4].x < landmarks[2].x
    else:
        thumb_open = landmarks[4].x > landmarks[2].x + 0.04
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    fingers = [thumb_open]
    for tip, pip in zip(finger_tips, finger_pips):
        fingers.append(landmarks[tip].y < landmarks[pip].y - 0.03)
    return sum(fingers)

def count_fingers_relaxed(landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    finger_states = []
    for tip_id in tip_ids:
        tip = landmarks[tip_id]
        pip = landmarks[tip_id - 2]
        if tip_id == 4:
            if hand_label == "Left":
                is_open = tip.x < pip.x
            else:
                is_open = tip.x > pip.x
        else:
            is_open = tip.y < pip.y
        finger_states.append(is_open)
    return finger_states.count(True)

def count_fingers(landmarks, hand_label):
    if current_mode == "NORMAL":
        return count_fingers_strict(landmarks, hand_label)
    else:
        return count_fingers_relaxed(landmarks, hand_label)

def is_fist(landmarks):
    tip_ids = [8, 12, 16, 20]
    for tip in tip_ids:
        if landmarks[tip].y < landmarks[tip - 2].y - 0.02:
            return False
    return True

def update_hand_data(landmarks, hand_label):
    # สลับ label มือเพื่อแก้ปัญหามือซ้ายขวาสลับกัน
    if hand_label == "Left":
        hand_label = "Right"
    elif hand_label == "Right":
        hand_label = "Left"

    hand_data = right_hand_data if hand_label == "Right" else left_hand_data
    hand_data.finger_count = count_fingers(landmarks, hand_label)
    hand_data.is_detected = True
    hand_data.landmarks = landmarks

def reset_hand_data():
    left_hand_data.is_detected = False
    right_hand_data.is_detected = False

def check_flash_control_trigger():
    # ถ้ามือขวากำมือแน่นในโหมด SYMBOL ให้สลับแฟลช
    if right_hand_data.is_detected and is_fist(right_hand_data.landmarks) and current_mode == "SYMBOL":
        return True
    return False

def check_mode_transitions(current_time, frame):
    global current_mode, mode_transition_start_time, mode_transition_grace_time, ok_hold_start_time, flash_on

    ok_symbol_detected = False
    if left_hand_data.is_detected and is_ok_symbol(left_hand_data.landmarks):
        ok_symbol_detected = True
    if right_hand_data.is_detected and is_ok_symbol(right_hand_data.landmarks):
        ok_symbol_detected = True

    # ถ้ามี OK symbol ตรวจจับทุกโหมด
    if ok_symbol_detected:
        if ok_hold_start_time is None:
            ok_hold_start_time = current_time
        elapsed = current_time - ok_hold_start_time
        remaining = OK_HOLD_DURATION - elapsed
        if remaining > 0:
            draw_ok_countdown(frame, remaining, OK_HOLD_DURATION)
        if elapsed >= OK_HOLD_DURATION:
            if current_mode == "FLASH_CONTROL" and flash_on:
                flash_off_request()
                flash_on = False
            current_mode = "NORMAL"
            print("[MODE] Switch to NORMAL mode")
            mode_transition_start_time = None
            mode_transition_grace_time = None
            ok_hold_start_time = None
        return
    else:
        ok_hold_start_time = None

    if current_mode == "NORMAL":
        open_hand_detected = ((left_hand_data.is_detected and left_hand_data.finger_count == 5) or
                              (right_hand_data.is_detected and right_hand_data.finger_count == 5))
        if open_hand_detected:
            if mode_transition_start_time is None:
                mode_transition_start_time = current_time
            mode_transition_grace_time = None
            elapsed = current_time - mode_transition_start_time
            if elapsed >= OPEN_HAND_DURATION:
                current_mode = "SYMBOL"
                print("[MODE] Switch to SYMBOL mode")
                mode_transition_start_time = None
                mode_transition_grace_time = None
        else:
            if mode_transition_start_time is not None:
                if mode_transition_grace_time is None:
                    mode_transition_grace_time = current_time
                elif (current_time - mode_transition_grace_time) > GRACE_PERIOD:
                    mode_transition_start_time = None
                    mode_transition_grace_time = None

    elif current_mode == "SYMBOL":
        if check_flash_control_trigger():
            current_mode = "FLASH_CONTROL"
            print("[MODE] Switch to FLASH_CONTROL mode")
        # ควบคุมแฟลชตามสถานะมือขวา
            if right_hand_data.finger_count == 0:  # กำมือ
                flash_off_request()
            elif right_hand_data.finger_count == 2:  # กางมือ
                flash_on_request()


    elif current_mode == "FLASH_CONTROL":
        # รอ OK symbol เพื่อกลับ NORMAL mode และปิดแฟลช (handled ข้างบน)
        pass

def draw_ok_countdown(frame, remaining, total):
    text = f"Hold OK: {int(remaining) + 1}s"
    cv2.putText(frame, text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

def draw_status_info(frame, current_time):
    cv2.putText(frame, f"Mode: {current_mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3)

    if ok_hold_start_time is not None:
        elapsed = current_time - ok_hold_start_time
        remaining = max(0, OK_HOLD_DURATION - elapsed)
        if remaining > 0:
            draw_ok_countdown(frame, remaining, OK_HOLD_DURATION)

    if mode_transition_start_time is not None:
        elapsed = current_time - mode_transition_start_time
        remaining = max(0, OPEN_HAND_DURATION - elapsed)
        if remaining > 0:
            text = f"Switching to SYMBOL in {int(remaining)+1}s"
            cv2.putText(frame, text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if left_hand_data.is_detected:
        cv2.putText(frame, f"Left fingers: {left_hand_data.finger_count}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if right_hand_data.is_detected:
        cv2.putText(frame, f"Right fingers: {right_hand_data.finger_count}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if current_mode == "FLASH_CONTROL":
        text = f"Flash: {'ON' if flash_on else 'OFF'}"
        cv2.putText(frame, text, (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

def main():
    print("[INFO] Starting ESP32-CAM Hand Gesture Control")
    cap = cv2.VideoCapture(CAM_STREAM_URL)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video stream from {CAM_STREAM_URL}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        reset_hand_data()

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label
                # สลับ label มือเพื่อแก้ปัญหามือซ้ายขวาสลับกัน
                if label == "Left":
                    label = "Right"
                elif label == "Right":
                    label = "Left"
                update_hand_data(hand_landmarks.landmark, label)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        current_time = time.time()
        check_mode_transitions(current_time, frame)
        draw_status_info(frame, current_time)

        cv2.imshow("ESP32-CAM Hand Gesture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] ESC pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
