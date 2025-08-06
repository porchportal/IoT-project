import cv2
import mediapipe as mp
import math
import time
from collections import deque
import requests
import numpy as np
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force use of xcb instead of wayland

# ====== CONFIG ======
ESP32_IP = "192.168.78.48"  # เปลี่ยนเป็น IP ของคุณ
FLASH_ON_URL = f"http://{ESP32_IP}/flash/on"
FLASH_OFF_URL = f"http://{ESP32_IP}/flash/off"
CAM_STREAM_URL = f"http://{ESP32_IP}/stream"

# ค่าคงที่สำหรับการควบคุมเสียง
MAX_DISTANCE = 0.35
MIN_DISTANCE = 0.02
DIST_HISTORY_LEN = 15  # ลดลงจาก 30
STABILITY_THRESHOLD = 0.01  # เพิ่มขึ้นเล็กน้อยเพื่อลดการคำนวณ

# ====== MediaPipe setup (ปรับแต่งสำหรับความเร็วและความเสถียร) ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,  # เพิ่มขึ้นเพื่อลดการตรวจจับผิด
    min_tracking_confidence=0.7,   # เพิ่มขึ้นเพื่อความเสถียร
    model_complexity=1             # ใช้โมเดลกลางระหว่างเร็วและแม่นยำ
)

# ====== โหมด ======
current_mode = "NORMAL"

# ====== เวลาและสถานะ ======
OPEN_HAND_DURATION = 1.5  # ลดลงจาก 2.0
OK_HOLD_DURATION = 0.8     # ลดลงจาก 1.0
GRACE_PERIOD = 0.3         # ลดลงจาก 0.4

mode_transition_start_time = None
ok_hold_start_time = None

# ====== สถานะมือ (ลดข้อมูลที่ไม่จำเป็น) ======
left_hand_data = {
    'finger_count': 0,
    'distance_history': deque(maxlen=DIST_HISTORY_LEN),
    'volume_level': 50,
    'is_detected': False,
    'landmarks': None
}

right_hand_data = {
    'finger_count': 0,
    'distance_history': deque(maxlen=DIST_HISTORY_LEN),
    'volume_level': 50,
    'is_detected': False,
    'landmarks': None
}

# ====== สถานะเพิ่มเติม ======
flash_on = False
frame_count = 0  # สำหรับ skip frames

# ====== ฟังก์ชันควบคุมแฟลช (เพิ่ม timeout สั้น) ======
def flash_control_request(turn_on):
    global flash_on
    if turn_on == flash_on:
        return

    try:
        url = FLASH_ON_URL if turn_on else FLASH_OFF_URL
        r = requests.get(url, timeout=2)  # ลด timeout จาก 5 เป็น 2
        if r.status_code == 200:
            flash_on = turn_on
    except:
        pass  # เพิ่ม silent fail เพื่อไม่ให้ช้า

# ====== ฟังก์ชันช่วย (ปรับปรุงความเร็ว) ======
def get_distance(lm1, lm2):
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    return math.sqrt(dx*dx + dy*dy)  # เร็วกว่าการยกกำลัง

def is_ok_symbol(landmarks):
    """Check if hand is making OK symbol (เร็วขึ้น)"""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    # คำนวณระยะห่างแบบง่าย
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    dist_sq = dx*dx + dy*dy  # ใช้ระยะห่างยกกำลังสองแทน sqrt
    
    if dist_sq > 0.0036:  # 0.06^2
        return False
    
    # เช็คนิ้วอื่นๆ แบบง่าย
    return (landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y)

def count_fingers_fast(landmarks, hand_label):
    """Fast finger counting (ผสม strict และ relaxed)"""
    count = 0
    
    # Thumb (simplified logic)
    if hand_label == "Right":
        if landmarks[4].x > landmarks[3].x:
            count += 1
    else:
        if landmarks[4].x < landmarks[3].x:
            count += 1
    
    # Other fingers (simplified)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y - 0.02:
            count += 1
    
    return count

def is_fist_fast(landmarks):
    """Fast fist detection"""
    # เช็คแค่ 2-3 นิ้วหลักก็พอ
    return (landmarks[8].y > landmarks[6].y + 0.02 and
            landmarks[12].y > landmarks[10].y + 0.02)

def is_gesture_stable_fast(distance_history):
    """Faster stability check"""
    if len(distance_history) < 5:  # ลดลงจาก 10
        return False
    
    recent = list(distance_history)[-5:]
    avg = sum(recent) / 5
    variance = sum((x - avg)**2 for x in recent) / 5
    return variance < STABILITY_THRESHOLD * STABILITY_THRESHOLD

# ====== ฟังก์ชันอัปเดตข้อมูลมือ (เร็วขึ้น) ======
def update_hand_data(landmarks, hand_label):
    hand_data = right_hand_data if hand_label == "Right" else left_hand_data
    
    hand_data['finger_count'] = count_fingers_fast(landmarks, hand_label)
    hand_data['is_detected'] = True
    hand_data['landmarks'] = landmarks
    
    # Volume control (เฉพาะเมื่อจำเป็น)
    if current_mode == "VOLUME_CONTROL" and hand_label == "Left":
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = get_distance(thumb_tip, index_tip)
        hand_data['distance_history'].append(distance)
        
        if is_gesture_stable_fast(hand_data['distance_history']):
            distance = max(MIN_DISTANCE, min(distance, MAX_DISTANCE))
            vol_percentage = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
            hand_data['volume_level'] = int(vol_percentage * 100)

def reset_hand_data():
    left_hand_data['is_detected'] = False
    right_hand_data['is_detected'] = False

# ====== ฟังก์ชันตรวจสอบการสลับโหมด (ปรับปรุง) ======
def check_mode_transitions(current_time):
    global current_mode, mode_transition_start_time, ok_hold_start_time, flash_on
    
    # Check OK symbol (เร็วขึ้น)
    ok_detected = False
    if ((left_hand_data['is_detected'] and is_ok_symbol(left_hand_data['landmarks'])) or
        (right_hand_data['is_detected'] and is_ok_symbol(right_hand_data['landmarks']))):
        ok_detected = True
    
    if ok_detected and current_mode != "NORMAL":
        if ok_hold_start_time is None:
            ok_hold_start_time = current_time
        elif current_time - ok_hold_start_time >= OK_HOLD_DURATION:
            if current_mode == "VOLUME_CONTROL":
                current_mode = "SYMBOL"
                left_hand_data['distance_history'].clear()
            elif current_mode == "SYMBOL":
                if flash_on:
                    flash_control_request(False)
                current_mode = "NORMAL"
            
            mode_transition_start_time = None
            ok_hold_start_time = None
        return
    else:
        ok_hold_start_time = None
    
    # Mode transitions (เร็วขึ้น)
    if current_mode == "NORMAL":
        open_hand = ((left_hand_data['is_detected'] and left_hand_data['finger_count'] == 5) or
                     (right_hand_data['is_detected'] and right_hand_data['finger_count'] == 5))
        
        if open_hand:
            if mode_transition_start_time is None:
                mode_transition_start_time = current_time
            elif current_time - mode_transition_start_time >= OPEN_HAND_DURATION:
                current_mode = "SYMBOL"
                mode_transition_start_time = None
        else:
            mode_transition_start_time = None

    elif current_mode == "SYMBOL":
        # Flash control (simplified)
        if right_hand_data['is_detected']:
            if is_fist_fast(right_hand_data['landmarks']) != flash_on:
                flash_control_request(not flash_on)
        
        # Switch to volume control
        if left_hand_data['is_detected'] and left_hand_data['finger_count'] == 5:
            if mode_transition_start_time is None:
                mode_transition_start_time = current_time
            elif current_time - mode_transition_start_time >= OPEN_HAND_DURATION:
                current_mode = "VOLUME_CONTROL"
                mode_transition_start_time = None
                left_hand_data['distance_history'].clear()
        else:
            mode_transition_start_time = None

    elif current_mode == "VOLUME_CONTROL":
        if right_hand_data['is_detected'] and right_hand_data['finger_count'] == 5:
            if mode_transition_start_time is None:
                mode_transition_start_time = current_time
            elif current_time - mode_transition_start_time >= OPEN_HAND_DURATION:
                current_mode = "SYMBOL"
                mode_transition_start_time = None
        else:
            mode_transition_start_time = None

# ====== ฟังก์ชันวาด UI (เร็วขึ้น) ======
def draw_text_fast(frame, text, pos, color=(255,255,255), scale=0.8):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def draw_status_info_fast(frame):
    # Mode display
    mode_colors = {
        "NORMAL": (0, 128, 255),
        "SYMBOL": (0, 255, 255),
        "VOLUME_CONTROL": (255, 255, 0)
    }
    color = mode_colors.get(current_mode, (255, 255, 255))
    draw_text_fast(frame, f"MODE: {current_mode}", (20, 40), color, 1.0)
    
    # Flash status (simplified)
    if current_mode in ["SYMBOL", "VOLUME_CONTROL"]:
        flash_text = "Flash: ON" if flash_on else "Flash: OFF"
        flash_color = (0, 255, 0) if flash_on else (0, 0, 255)
        draw_text_fast(frame, flash_text, (frame.shape[1] - 150, 40), flash_color)
    
    # Finger counts (เฉพาะเมื่อตรวจพบ)
    y_pos = 80
    if left_hand_data['is_detected']:
        draw_text_fast(frame, f"L: {left_hand_data['finger_count']}", (20, y_pos), (255, 255, 255))
        y_pos += 30
    if right_hand_data['is_detected']:
        draw_text_fast(frame, f"R: {right_hand_data['finger_count']}", (20, y_pos), (255, 255, 255))
    
    # Volume (เฉพาะในโหมด Volume Control)
    if current_mode == "VOLUME_CONTROL" and left_hand_data['is_detected']:
        vol = left_hand_data['volume_level']
        draw_text_fast(frame, f"Volume: {vol}%", (20, frame.shape[0] - 40), (255, 255, 0))

# ====== ฟังก์ชันหลัก (เพิ่ม FPS optimization) ======
def main():
    global frame_count
    
    print("[INFO] Starting Optimized Hand Gesture Control")
    
    # Camera setup with optimization
    cap = cv2.VideoCapture(CAM_STREAM_URL)
    if not cap.isOpened():
        print("[INFO] ESP32 stream not available, trying local camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[INFO] Optimized Controls:")
    print("  - ESC: Exit")
    print("  - F: Manual flash toggle")

    results = None
    process_frame = True  # Control when to process frames
    skip_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        skip_counter += 1
        
        # Process every 3rd frame instead of every 2nd for better stability
        if skip_counter >= 3:
            skip_counter = 0
            process_frame = True
        else:
            process_frame = False
        
        if process_frame:
            # Use original frame size for better accuracy, but process less frequently
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            reset_hand_data()
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = hand_handedness.classification[0].label
                    update_hand_data(hand_landmarks.landmark, label)
            
            check_mode_transitions(time.time())
        
        # Always draw landmarks from the last successful detection for smooth display
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw with better styling for smoother appearance
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        draw_status_info_fast(frame)
        
        # Show frame with improved window handling
        cv2.imshow("Fast Hand Gesture Control", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('f') or key == ord('F'):
            flash_control_request(not flash_on)

    cap.release()
    cv2.destroyAllWindows()
    
    if flash_on:
        flash_control_request(False)
    
    print("[INFO] Program exited.")

if __name__ == "__main__":
    main()