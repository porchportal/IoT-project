import cv2
import mediapipe as mp
import math
import requests
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)

# ESP32-CAM IP ที่รัน server (แก้ตามจริง)
ESP32_IP = "192.168.78.48"

def flash_control(turn_on: bool):
    try:
        url = f"http://{ESP32_IP}/flash/on" if turn_on else f"http://{ESP32_IP}/flash/off"
        r = requests.get(url, timeout=0.5)
        print(f"Flash {'ON' if turn_on else 'OFF'} response:", r.status_code)
    except Exception as e:
        print("Error sending flash control:", e)

def fingers_up(landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    for i, tip_id in enumerate(tip_ids):
        tip = landmarks[tip_id]
        pip = landmarks[tip_id - 2]
        if i == 0:  # thumb
            if hand_label == "Left":
                fingers.append(tip.x < pip.x)
            else:
                fingers.append(tip.x > pip.x)
        else:
            fingers.append(tip.y < pip.y)
    return fingers

def is_ok_sign(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    touching = dist < 0.06
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    return touching and middle_up and ring_up and pinky_up

flash_on = False
current_mode = "NORMAL"
volume_level = 50
switch_mode_flag = False

cap = cv2.VideoCapture(0)  # เปลี่ยนเลข 0 เป็นกล้องที่ใช้

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    left_fingers = [False]*5
    right_fingers = [False]*5
    left_ok = False

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label  # Left or Right
            lm = hand_landmarks.landmark

            fingers = fingers_up(lm, label)

            if label == "Left":
                left_fingers = fingers
                left_ok = is_ok_sign(lm)
            else:
                right_fingers = fingers

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Feature 1-2: Flash control by left hand
    if left_fingers[1]:  # นิ้วชี้เปิดแฟลช
        if not flash_on:
            flash_control(True)
            flash_on = True
    elif left_fingers[0]:  # นิ้วโป้งปิดแฟลช
        if flash_on:
            flash_control(False)
            flash_on = False

    # Feature 3: Stream trigger (right hand middle finger)
    if right_fingers[2]:
        cv2.putText(frame, "Streaming triggered", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        # TODO: สั่งงานสตรีมหรือถ่ายภาพที่นี่

    # Feature 4: Switch mode (right hand index + middle)
    if right_fingers[1] and right_fingers[2]:
        if not switch_mode_flag:
            current_mode = "VOLUME_CONTROL" if current_mode == "NORMAL" else "NORMAL"
            print(f"Mode switched to {current_mode}")
            switch_mode_flag = True
    else:
        switch_mode_flag = False

    # Feature 5: Reset mode (left hand full open)
    if all(left_fingers):
        current_mode = "NORMAL"
        print("Mode reset to NORMAL")

    # Feature 6-7: Volume up/down (right hand ring/pinky)
    if right_fingers[3]:
        volume_level = min(100, volume_level + 1)
    if right_fingers[4]:
        volume_level = max(0, volume_level - 1)

    # Feature 8: Volume control mode (right thumb + index)
    if right_fingers[0] and right_fingers[1]:
        current_mode = "VOLUME_CONTROL"

    # Feature 9: Symbol mode (right middle + ring)
    if right_fingers[2] and right_fingers[3]:
        current_mode = "SYMBOL"

    # Feature 10: OK sign left hand
    if left_ok:
        current_mode = "NORMAL"
        print("OK sign detected: reset to NORMAL")

    # แสดงสถานะ
    cv2.putText(frame, f"Flash: {'ON' if flash_on else 'OFF'}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Mode: {current_mode}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, f"Volume: {volume_level}%", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Control ESP32-CAM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
