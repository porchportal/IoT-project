import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
import requests
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
from concurrent.futures import ThreadPoolExecutor

# Add this after your ESP32 configuration section
# ====== ASYNC LOGGING SETUP ======
log_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="LogSender")

# ====== ESP32 CONFIGURATION ======
ESP32_IP = "172.20.10.3"  # Change to your ESP32 IP
FLASH_ON_URL = f"http://{ESP32_IP}/flash/on"
FLASH_OFF_URL = f"http://{ESP32_IP}/flash/off"
log_url = f"http://{ESP32_IP}/receive"
# CAM_STREAM_URL = f"http://{ESP32_IP}/stream"
# http://172.20.10.3/receive

# ====== MediaPipe Setup (Optimized) ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# ====== SHARED MODE STATE ======
current_mode = "NORMAL"  # NORMAL → SYMBOL → VOLUME_CONTROL
mode_transition_start_time = None
mode_transition_grace_time = None

# ====== TIMING CONSTANTS ======
GRACE_PERIOD = 0.3
OPEN_HAND_DURATION = 1.5
FINGER_HOLD_DURATION = 0.8
VOLUME_HOLD_DURATION = 2.0
OK_HOLD_DURATION = 0.8

# ====== VOLUME & FLASH CONTROL ======
MAX_DISTANCE = 0.35
MIN_DISTANCE = 0.02
DIST_HISTORY_LEN = 15
STABILITY_THRESHOLD = 0.008
flash_on = False
FLASH_ON_THRESHOLD = 60   # Volume > 60 turns flash ON
FLASH_OFF_THRESHOLD = 20  # Volume < 20 turns flash OFF

# ====== GESTURE TRACKING ======
finger_hold_start_time = None
ok_hold_start_time = None
saved_number = None
saved_hand = None
PENDING_LOG_HOLD_SEC = 2.0

volume_mode_entry_hand = None
volume_mode_entry_count = None

# ====== SEPARATE HAND DATA ======
left_hand_data = {
    'finger_count': 0,
    'volume_level': 50,
    'distance_history': deque(maxlen=DIST_HISTORY_LEN),
    'volume_hold_start_time': None,
    'saved_data': [],
    'is_detected': False,
    'landmarks': None,
    'stable_start_time': None,
    'has_sent_log': False,
}

right_hand_data = {
    'finger_count': 0,
    'volume_level': 50,
    'distance_history': deque(maxlen=DIST_HISTORY_LEN),
    'volume_hold_start_time': None,
    'saved_data': [],
    'is_detected': False,
    'landmarks': None,
    'stable_start_time': None,
    'has_sent_log': False,
}

# ====== PERFORMANCE TRACKING ======
frame_count = 0
skip_counter = 0

# ====== FIXED LOGGING SYSTEM ======
def update_stability_and_log():
    """Update stability tracking and send logs for stable gestures"""
    now = time.time()
    
    for hand_data, label in zip([left_hand_data, right_hand_data], ["Left", "Right"]):
        if hand_data['is_detected'] and current_mode == "VOLUME_CONTROL":
            # Check if gesture is stable
            if is_gesture_stable(hand_data['distance_history']):
                # Start stability timer if not already started
                if hand_data.get('stable_start_time') is None:
                    hand_data['stable_start_time'] = now
                    hand_data['has_sent_log'] = False
                    print(f"✓ {label} hand gesture becoming stable...")
                
                # Check if we've been stable long enough and haven't sent log yet
                elif not hand_data.get('has_sent_log', False):
                    stable_duration = now - hand_data['stable_start_time']
                    if stable_duration >= PENDING_LOG_HOLD_SEC:  # 2 seconds stable
                        # Send log for current stable gesture
                        if current_mode == "VOLUME_CONTROL" and label == volume_mode_entry_hand:
                            # Use entry finger count for Volume mode
                            send_log_to_esp32(volume_mode_entry_count, flash_on, label)
                        else:
                            send_log_to_esp32(hand_data['finger_count'], flash_on, label)
                        hand_data['has_sent_log'] = True
                        print(f"📤 {label} hand log sent after {stable_duration:.1f}s stability")
            else:
                # Gesture not stable, reset stability tracking
                # if hand_data.get('stable_start_time') is not None:
                #     print(f"⚠️ {label} hand gesture became unstable, resetting timer")
                hand_data['stable_start_time'] = None
                hand_data['has_sent_log'] = False
        else:
            # Hand not detected, reset everything
            hand_data['stable_start_time'] = None
            hand_data['has_sent_log'] = False

# ====== ESP32 FLASH CONTROL FUNCTIONS ======
def flash_control_request(turn_on):
    """Control ESP32 flash with timeout and error handling (FIXED - removed logging logic)"""
    global flash_on
    if turn_on == flash_on:
        return
    
    try:
        url = FLASH_ON_URL if turn_on else FLASH_OFF_URL
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            flash_on = turn_on
            print(f"💡 Flash {'ON' if turn_on else 'OFF'}")
    except requests.RequestException as e:
        print(f"❌ Flash control error: {e}")

def send_log_to_esp32(finger_count, flash_state, hand_label):
    """Non-blocking version - submits to background thread"""
    log_data = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "finger_count": finger_count,
        "flash_state": "ON" if flash_state else "OFF",
        "hand_label": hand_label
    }
    
    # Submit to thread pool - returns immediately, doesn't block main thread
    log_executor.submit(_send_log_worker, log_data)

def _send_log_worker(log_data):
    """Background worker - runs in separate thread"""
    try:
        response = requests.post(log_url, json=log_data, timeout=10)
        print(f"✓ Log sent: {log_data['hand_label']} hand - {log_data['finger_count']} fingers")
    except Exception as e:
        print(f"✗ Log failed: {e}")

def control_flash_by_volume(volume):
    """Control flash based on volume level with hysteresis"""
    global flash_on
    
    if volume > FLASH_ON_THRESHOLD and not flash_on:
        flash_control_request(True)
    elif volume < FLASH_OFF_THRESHOLD and flash_on:
        flash_control_request(False)

# ====== GESTURE DETECTION FUNCTIONS ======
def get_distance(lm1, lm2):
    """Calculate distance between two landmarks (optimized)"""
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    return math.sqrt(dx*dx + dy*dy)

def is_ok_symbol(landmarks):
    """Check if hand is making OK symbol (optimized)"""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    # Fast distance calculation
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    dist_sq = dx*dx + dy*dy
    
    if dist_sq > 0.0036:  # 0.06^2
        return False
    
    # Check other fingers are up
    return (landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y)

def count_fingers_strict(landmarks, hand_label):
    """Strict finger counting for NORMAL mode"""
    count = 0
    
    # Thumb logic
    if hand_label == "Right":
        if landmarks[4].x < landmarks[2].x:
            count += 1
    else:
        if landmarks[4].x > landmarks[2].x + 0.04:
            count += 1
    
    # Other fingers
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y - 0.03:
            count += 1
    
    return count

def count_fingers_relaxed(landmarks, hand_label):
    """Relaxed finger counting for SYMBOL and VOLUME_CONTROL modes"""
    count = 0
    
    # Thumb logic
    if hand_label == "Right":
        if landmarks[4].x > landmarks[4-2].x:
            count += 1
    else:
        if landmarks[4].x < landmarks[4-2].x:
            count += 1
    
    # Other fingers
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    
    return count

def count_fingers(landmarks, hand_label):
    """Dynamically choose finger counting method based on mode"""
    if current_mode == "NORMAL":
        return count_fingers_strict(landmarks, hand_label)
    else:
        return count_fingers_relaxed(landmarks, hand_label)

def is_open_hand(landmarks, hand_label):
    """Check if hand is fully open (all 5 fingers extended)"""
    return count_fingers(landmarks, hand_label) == 5

def is_gesture_stable(distance_history):
    """Check if the current gesture is stable (IMPROVED)"""
    if len(distance_history) < 5:  # Reduced minimum requirement
        return False
    
    # Use more recent samples for faster response
    recent_distances = list(distance_history)[-8:]
    if len(recent_distances) < 5:
        return False
        
    std_deviation = np.std(recent_distances)
    return std_deviation < STABILITY_THRESHOLD

def get_volume_message(volume):
    """Generate a descriptive message based on volume level"""
    if volume <= 10:
        return "Close"
    elif volume <= 30:
        return "Quiet Environment"
    elif volume <= 50:
        return "Normal Level"
    elif volume <= 70:
        return "Loud Environment"
    elif volume <= 90:
        return "Very Loud"
    else:
        return "Maximum Volume"

def control_volume_for_hand(landmarks, hand_data):
    """Control volume for individual hand and update flash based on volume"""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = get_distance(thumb_tip, index_tip)
    hand_data['distance_history'].append(distance)
    
    # Convert distance to volume
    distance = max(MIN_DISTANCE, min(distance, MAX_DISTANCE))
    vol_percentage = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    volume = int(vol_percentage * 100)
    
    # Control flash based on volume level
    control_flash_by_volume(volume)
    
    return volume

# ====== HAND DATA UPDATE FUNCTIONS ======
def update_hand_data(landmarks, hand_side):
    """Update individual hand data (ENHANCED with distance tracking)"""
    hand_data = right_hand_data if hand_side == "Right" else left_hand_data
    
    # Always update finger count
    hand_data['finger_count'] = count_fingers(landmarks, hand_side)
    hand_data['is_detected'] = True
    hand_data['landmarks'] = landmarks
    
    # Update distance history for stability tracking (using thumb-index distance)
    if current_mode in ["SYMBOL", "VOLUME_CONTROL"]:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = get_distance(thumb_tip, index_tip)
        hand_data['distance_history'].append(distance)
    
    # Update volume control if in that mode
    if current_mode == "VOLUME_CONTROL":
        hand_data['volume_level'] = control_volume_for_hand(landmarks, hand_data)
    
    return hand_data

def reset_hand_data():
    """Reset hand detection flags"""
    left_hand_data['is_detected'] = False
    right_hand_data['is_detected'] = False

# ====== MODE TRANSITION FUNCTIONS ======
def check_mode_transitions(current_time):
    """Check for mode transitions based on gestures from either hand"""
    global current_mode, mode_transition_start_time, finger_hold_start_time
    global mode_transition_grace_time, ok_hold_start_time, saved_number, saved_hand
    global volume_mode_entry_hand, volume_mode_entry_count
    volume_mode_entry_hand = saved_hand
    volume_mode_entry_count = saved_number
    
    # Check for OK symbol from either hand to go back
    ok_symbol_detected = False
    if left_hand_data['is_detected'] and is_ok_symbol(left_hand_data['landmarks']):
        ok_symbol_detected = True
    if right_hand_data['is_detected'] and is_ok_symbol(right_hand_data['landmarks']):
        ok_symbol_detected = True
    
    if ok_symbol_detected and current_mode != "NORMAL":
        if ok_hold_start_time is None:
            ok_hold_start_time = current_time
        elapsed = current_time - ok_hold_start_time
        if elapsed >= OK_HOLD_DURATION:
            if current_mode == "VOLUME_CONTROL":
                current_mode = "SYMBOL"
                left_hand_data['distance_history'].clear()
                right_hand_data['distance_history'].clear()
                print("OK symbol held! Switching back to SYMBOL mode.")
            elif current_mode == "SYMBOL":
                current_mode = "NORMAL"
                saved_number = None
                print("OK symbol held! Switching back to NORMAL mode.")
            
            mode_transition_start_time = None
            finger_hold_start_time = None
            ok_hold_start_time = None
        return
    else:
        ok_hold_start_time = None
    
    # NORMAL mode: detect open hand to switch to SYMBOL mode
    if current_mode == "NORMAL":
        open_hand_detected = (
            (left_hand_data['is_detected'] and left_hand_data['finger_count'] == 5) or
            (right_hand_data['is_detected'] and right_hand_data['finger_count'] == 5)
        )
        
        if open_hand_detected:
            if mode_transition_start_time is None:
                mode_transition_start_time = current_time
            mode_transition_grace_time = None
            elapsed = current_time - mode_transition_start_time
            if elapsed >= OPEN_HAND_DURATION:
                current_mode = "SYMBOL"
                mode_transition_start_time = None
                print("Symbol Mode Activated!")
        else:
            if mode_transition_start_time is not None:
                if mode_transition_grace_time is None:
                    mode_transition_grace_time = current_time
                elif current_time - mode_transition_grace_time > GRACE_PERIOD:
                    mode_transition_start_time = None
                    mode_transition_grace_time = None
    
    # SYMBOL mode: check for finger gestures to enter VOLUME_CONTROL
    elif current_mode == "SYMBOL":
        left_count = left_hand_data['finger_count']
        right_count = right_hand_data['finger_count']
        
        left_valid = left_hand_data['is_detected'] and (1 <= left_count <= 5)
        right_valid = right_hand_data['is_detected'] and (1 <= right_count <= 5)
        
        if left_valid:
            active_hand = "Left"
            active_count = left_count
        elif right_valid:
            active_hand = "Right"
            active_count = right_count
        else:
            active_hand = None
            active_count = None
        
        if active_hand is not None:
            if (finger_hold_start_time is None or saved_number != active_count 
                or saved_hand != active_hand):
                finger_hold_start_time = current_time
                saved_number = active_count
                saved_hand = active_hand
            elapsed = current_time - finger_hold_start_time
            if elapsed >= FINGER_HOLD_DURATION:
                current_mode = "VOLUME_CONTROL"
                finger_hold_start_time = None
                left_hand_data['distance_history'].clear()
                right_hand_data['distance_history'].clear()
                print(f"Volume Control Mode activated! ({saved_hand} hand, {saved_number} fingers)")
        else:
            finger_hold_start_time = None
            saved_number = None
            saved_hand = None

# ====== DRAWING FUNCTIONS ======
def draw_text_fast(frame, text, pos, color=(255, 255, 255), scale=0.8):
    """Fast text drawing"""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def draw_volume_bar(frame, volume, x, y):
    """Draw volume bar with flash indicator"""
    bar_width, bar_height = 40, 200
    
    # Draw outer rectangle
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)
    
    # Draw volume fill
    fill_height = int((volume / 100) * bar_height)
    fill_y = y + bar_height - fill_height
    
    # Color based on volume level and flash thresholds
    if volume < FLASH_OFF_THRESHOLD:
        color = (0, 255, 0)  # Green - Flash OFF zone
    elif volume > FLASH_ON_THRESHOLD:
        color = (0, 0, 255)  # Red - Flash ON zone
    else:
        color = (0, 255, 255)  # Yellow - Neutral zone
    
    cv2.rectangle(frame, (x + 2, fill_y), (x + bar_width - 2, y + bar_height - 2), color, -1)
    
    # Add volume percentage text
    cv2.putText(frame, f"{volume}%", (x - 5, y + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add flash threshold indicators
    flash_on_y = y + bar_height - int((FLASH_ON_THRESHOLD / 100) * bar_height)
    flash_off_y = y + bar_height - int((FLASH_OFF_THRESHOLD / 100) * bar_height)
    
    cv2.line(frame, (x - 5, flash_on_y), (x + bar_width + 5, flash_on_y), (0, 0, 255), 2)
    cv2.line(frame, (x - 5, flash_off_y), (x + bar_width + 5, flash_off_y), (0, 255, 0), 2)
    
    # Labels
    cv2.putText(frame, "ON", (x + bar_width + 10, flash_on_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, "OFF", (x + bar_width + 10, flash_off_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

def draw_countdown_circle(frame, remaining_time, total_time, center=(320, 240)):
    """Draw circular countdown timer"""
    radius = 50
    angle = (remaining_time / total_time) * 2 * math.pi
    
    # Background circle
    cv2.circle(frame, center, radius, (50, 50, 50), -1)
    cv2.circle(frame, center, radius, (255, 255, 255), 3)
    
    # Countdown arc
    if remaining_time > 0:
        start_angle = -math.pi / 2
        end_angle = start_angle + angle
        points = [center]
        for a in np.linspace(start_angle, end_angle, 30):
            x = int(center[0] + radius * math.cos(a))
            y = int(center[1] + radius * math.sin(a))
            points.append((x, y))
        if len(points) > 2:
            cv2.fillPoly(frame, [np.array(points)], (0, 255, 0))
    
    # Countdown number
    countdown_num = int(remaining_time) + 1
    text_size = cv2.getTextSize(str(countdown_num), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(frame, str(countdown_num), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

def draw_stability_countdowns(frame):
    """Draw countdown circles for both hands when they're stable"""
    now = time.time()
    
    # Left hand countdown
    if (left_hand_data['is_detected'] and 
        is_gesture_stable(left_hand_data['distance_history']) and 
        left_hand_data.get('stable_start_time') is not None and 
        not left_hand_data.get('has_sent_log', False)):
        
        elapsed = now - left_hand_data['stable_start_time']
        if elapsed < 2.0:
            remaining = 2.0 - elapsed
            draw_countdown_circle(frame, remaining, 2.0, center=(150, 400))
            # Add label
            cv2.putText(frame, "LEFT", (125, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Right hand countdown  
    if (right_hand_data['is_detected'] and 
        is_gesture_stable(right_hand_data['distance_history']) and 
        right_hand_data.get('stable_start_time') is not None and 
        not right_hand_data.get('has_sent_log', False)):
        
        elapsed = now - right_hand_data['stable_start_time']
        if elapsed < 2.0:
            remaining = 2.0 - elapsed
            draw_countdown_circle(frame, remaining, 2.0, center=(490, 400))
            # Add label
            cv2.putText(frame, "RIGHT", (465, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def draw_interface(frame, current_time):
    """Draw main interface"""
    # Mode indicator
    mode_colors = {
        "NORMAL": (100, 100, 100),
        "SYMBOL": (0, 150, 0),
        "VOLUME_CONTROL": (150, 150, 0)
    }
    cv2.rectangle(frame, (200, 5), (450, 45), mode_colors[current_mode], -1)
    cv2.putText(frame, f"MODE: {current_mode}", (210, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Flash status
    flash_status = "FLASH: ON" if flash_on else "FLASH: OFF"
    flash_color = (0, 255, 0) if flash_on else (0, 0, 255)
    cv2.putText(frame, flash_status, (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, flash_color, 2)
    
    # Hand information
    y_offset = 70
    if left_hand_data['is_detected']:
        cv2.putText(frame, f"LEFT: {left_hand_data['finger_count']} fingers", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if current_mode == "VOLUME_CONTROL":
            volume_msg = get_volume_message(left_hand_data['volume_level'])
            cv2.putText(frame, f"Vol: {left_hand_data['volume_level']}% - {volume_msg}",
                        (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            draw_volume_bar(frame, left_hand_data['volume_level'], 20, 150)
        y_offset += 50

    
    if right_hand_data['is_detected']:
        cv2.putText(frame, f"RIGHT: {right_hand_data['finger_count']} fingers", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if current_mode == "VOLUME_CONTROL":
            volume_msg = get_volume_message(right_hand_data['volume_level'])
            cv2.putText(frame, f"Vol: {right_hand_data['volume_level']}% - {volume_msg}",
                        (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            draw_volume_bar(frame, right_hand_data['volume_level'], 450, 150)
    
    # Transition progress
    if current_mode == "NORMAL" and mode_transition_start_time is not None:
        elapsed = current_time - mode_transition_start_time
        remaining = OPEN_HAND_DURATION - elapsed
        progress = elapsed / OPEN_HAND_DURATION
        bar_width = int(300 * progress)
        cv2.rectangle(frame, (160, 450), (480, 480), (50, 50, 50), -1)
        cv2.rectangle(frame, (160, 450), (160 + bar_width, 480), (0, 255, 255), -1)
        cv2.rectangle(frame, (160, 450), (480, 480), (255, 255, 255), 2)
        cv2.putText(frame, f"Hold open hand: {remaining:.1f}s", (160, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if current_mode == "SYMBOL" and finger_hold_start_time is not None:
        elapsed = current_time - finger_hold_start_time
        remaining = FINGER_HOLD_DURATION - elapsed
        draw_countdown_circle(frame, remaining, FINGER_HOLD_DURATION)
        cv2.putText(frame, "Activating Volume Control...", (200, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    if ok_hold_start_time is not None:
        elapsed = current_time - ok_hold_start_time
        remaining = OK_HOLD_DURATION - elapsed
        cv2.putText(frame, f"Hold OK: {remaining:.1f}s", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Instructions
    instructions = {
        "NORMAL": "Show open hand (either) for 1.5s to enter Symbol mode",
        "SYMBOL": "Show 1-5 fingers and hold for 0.8s to enter Volume Control",
        "VOLUME_CONTROL": "Control volume with pinch gesture - Flash controlled by volume level"
    }
    cv2.putText(frame, instructions[current_mode], (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ====== MAIN FUNCTION ======
def main():
    global frame_count, skip_counter
    
    print("=== FIXED GESTURE LOGGING SYSTEM ===")
    print(f"ESP32 IP: {ESP32_IP}")
    print("Logs will be sent after 2 seconds of stable gesture")
    print("Flash Control: Volume > 60% = ON, Volume < 20% = OFF")
    print("Controls:")
    print("  - ESC: Exit")
    print("  - R: Reset to Normal mode")
    print("  - F: Manual flash toggle")
    
    # Camera setup
    # cap = cv2.VideoCapture(CAM_STREAM_URL)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ESP32 stream not available, trying local camera...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Cannot open camera")
            return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 18)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    results = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        skip_counter += 1
        
        # Process every 2nd frame for better performance
        if skip_counter >= 2:
            skip_counter = 0
            
            # frame = cv2.flip(frame, 1)
            # frame = cv2.flip(frame, 0)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            reset_hand_data()
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    landmarks = hand_landmarks.landmark
                    update_hand_data(landmarks, label)
            
            check_mode_transitions(time.time())
            update_stability_and_log()
        
        # Always draw landmarks and interface
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        draw_interface(frame, time.time())
        draw_stability_countdowns(frame)
        
        cv2.imshow('Volume-Controlled Flash Gesture System', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            current_mode = "NORMAL"
            mode_transition_start_time = None
            finger_hold_start_time = None
            ok_hold_start_time = None
            saved_number = None
            saved_hand = None
            left_hand_data['distance_history'].clear()
            right_hand_data['distance_history'].clear()
            if flash_on:
                flash_control_request(False)
            print("Reset to Normal Mode")
        elif key == ord('f') or key == ord('F'):
            flash_control_request(not flash_on)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Turn off flash when exiting
    if flash_on:
        flash_control_request(False)
    
    print("Program exited.")

if __name__ == "__main__":
    main()
