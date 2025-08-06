import cv2
import mediapipe as mp
import math
import requests
import time

ESP32_IP = "192.168.78.48"  # เปลี่ยน IP ตามจริง
ESP32_STREAM_URL = f"http://{ESP32_IP}:81/stream"

def flash_control(turn_on: bool):
    try:
        url = f"http://{ESP32_IP}/flash/on" if turn_on else f"http://{ESP32_IP}/flash/off"
        r = requests.get(url, timeout=0.5)
        print(f"Flash {'ON' if turn_on else 'OFF'} response:", r.status_code)
    except Exception as e:
        print("Error sending flash control:", e)

def check_esp32_cam_available():
    try:
        test = requests.get(ESP32_STREAM_URL, timeout=1)
        return test.status_code == 200
    except:
        return False

# ตรวจสอบว่ามีกล้อง ESP32-CAM ใช้ได้ไหม
use_esp32 = check_esp32_cam_available()
print("ESP32-CAM found:", use_esp32)

if use_esp32:
    cap = cv2.VideoCapture(ESP32_STREAM_URL)
else:
    cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)

def fingers_up(lm, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    for i, tip_id in enumerate(tip_ids):
        tip = lm[tip_id]
        pip = lm[tip_id - 2]
        if i == 0:
            if hand_label == "Left":
                fingers.append(tip.x < pip.x)
            else:
                fingers.append(tip.x > pip.x)
        else:
            fingers.append(tip.y < pip.y)
    return fingers

def is_ok_sign(lm):
    thumb_tip = lm[4]
    index_tip = lm[8]
    dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    touching = dist < 0.06
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y
    return touching and middle_up and ring_up and pinky_up

flash_on = False
mode = "NORMAL"
switch_flag = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    left_fingers = [False]*5
    right_fingers = [False]*5
    left_ok = False

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            lm = hand_landmarks.landmark
            fingers = fingers_up(lm, label)

            if label == "Left":
                left_fingers = fingers
                left_ok = is_ok_sign(lm)
            else:
                right_fingers = fingers

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Feature 1: เปิดแฟลช ด้วย นิ้วชี้ซ้าย
    if left_fingers[1] and not flash_on:
        flash_control(True)
        flash_on = True

    # Feature 2: ปิดแฟลช ด้วย นิ้วโป้งซ้าย
    if left_fingers[0] and flash_on:
        flash_control(False)
        flash_on = False

    # Feature 3: ท่าชูนิ้วกลางขวา (สตรีมภาพ)
    if right_fingers[2]:
        cv2.putText(frame, "Feature 3: Streaming Trigger", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # Feature 4: สลับโหมดด้วย นิ้วชี้+กลางขวา
    if right_fingers[1] and right_fingers[2]:
        if not switch_flag:
            mode = "VOLUME" if mode == "NORMAL" else "NORMAL"
            print(f"Switched mode to {mode}")
            switch_flag = True
    else:
        switch_flag = False

    # Feature 5: รีเซ็ตโหมด ด้วย มือซ้ายกางนิ้วเต็ม
    if all(left_fingers):
        mode = "NORMAL"
        print("Reset mode to NORMAL")

    # Feature 6: เพิ่มเสียง (วงแหวนขวาขึ้น)
    if right_fingers[3]:
        cv2.putText(frame, "Feature 6: Volume Up", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Feature 7: ลดเสียง (ก้อยขวาขึ้น)
    if right_fingers[4]:
        cv2.putText(frame, "Feature 7: Volume Down", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Feature 8: โหมดเสียง ด้วย นิ้วโป้ง+ชี้ขวา
    if right_fingers[0] and right_fingers[1]:
        mode = "VOLUME_CONTROL"

    # Feature 9: โหมดสัญลักษณ์ ด้วย นิ้วกลาง+วงแหวนขวา
    if right_fingers[2] and right_fingers[3]:
        mode = "SYMBOL"

    # Feature 10: ท่า OK ซ้าย เพื่อรีเซ็ตโหมด
    if left_ok:
        mode = "NORMAL"
        print("OK sign detected, reset mode")

    # แสดงสถานะ
    cv2.putText(frame, f"Flash: {'ON' if flash_on else 'OFF'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Mode: {mode}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    if not use_esp32:
        cv2.putText(frame, "ESP32-CAM not found. Using local camera.", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("ESP32-CAM Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()