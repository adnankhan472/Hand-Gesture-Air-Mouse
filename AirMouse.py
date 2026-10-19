import os
import math
import time
import urllib.request

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# =====================  SETTINGS  =====================
CAM_INDEX = 0              # change to 1 if the wrong camera opens
CAM_W, CAM_H = 640, 480

frame_reduction = 100      # bigger = less hand movement needed to cross the screen

# Cursor smoothing
min_smooth = 2             # lower = faster response when moving quickly
max_smooth = 15            # higher = steadier when moving slowly
jitter_dist = 3            # screen pixels; smaller moves are ignored

# Scrolling
scroll_speed = 20
scroll_line_y = 150        # purple line (pixels from top of video)
scroll_dead_zone = 20      # no scroll within this many pixels of the line

# Clicks
pinch_ratio = 0.25         # pinch gap / palm size. Bigger = easier to trigger
click_cooldown = 0.5
right_click_cooldown = 1.0
right_fist_hold = 0.3      # seconds a right fist must be held before right click
safety_delay = 2.0         # seconds left click is blocked after a right click

# Exit gesture
exit_hold_seconds = 5
# ======================================================

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ---------- Model file (downloaded once) ----------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (one time only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise SystemExit(f"Could not download the model: {e}\n"
                         f"Download it manually from:\n{MODEL_URL}\n"
                         f"and save it next to this script as hand_landmarker.task")
    print("Download complete.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ---------- Camera and detector ----------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit("Could not open the camera. Close other apps using it, "
                     "or change CAM_INDEX at the top of the file.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
hand_detector = vision.HandLandmarker.create_from_options(options)
screen_w, screen_h = pyautogui.size()

# ---------- State ----------
prev_x, prev_y = pyautogui.position()
moving = False                 # was the cursor being moved on the previous frame?
last_left_click = 0.0
last_right_click = 0.0
right_fist_since = 0.0
exit_start = 0.0
last_timestamp_ms = 0
start_time = time.time()
prev_frame_time = time.time()


# ---------- Helpers ----------
def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def to_pixels(landmarks, w, h):
    """Converts the 21 normalized landmarks to pixel positions."""
    return [(lm.x * w, lm.y * h) for lm in landmarks]


def finger_states(pts):
    """Returns [thumb, index, middle, ring, pinky] as True (extended) / False (folded).
    A finger is extended when its tip is farther from the wrist than its middle joint.
    This works even if the hand is tilted or rotated."""
    wrist = pts[0]
    states = [dist(pts[4], pts[17]) > dist(pts[3], pts[17])]          # thumb
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        states.append(dist(pts[tip], wrist) > dist(pts[pip], wrist))
    return states


def is_fist(states):
    return not any(states[1:])          # all four fingers folded


def draw_hand(frame, pts):
    ipts = [(int(x), int(y)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, ipts[a], ipts[b], (0, 255, 0), 2)
    for p in ipts:
        cv2.circle(frame, p, 4, (0, 0, 255), cv2.FILLED)


print("System Ready.")
print("LEFT hand : index finger = move | index + middle = scroll")
print("RIGHT hand: pinch = left click  | fist = right click")
print("Exit      : both fists for 5 seconds, or press 'q' in the video window")

# ---------- Main loop ----------
while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera frame could not be read. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    now = time.time()

    # Send frame to MediaPipe (timestamps must always increase)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int((now - start_time) * 1000)
    if ts <= last_timestamp_ms:
        ts = last_timestamp_ms + 1
    last_timestamp_ms = ts
    result = hand_detector.detect_for_video(mp_image, ts)

    # Pre-compute everything about each detected hand
    hands = []
    for i, lms in enumerate(result.hand_landmarks):
        pts = to_pixels(lms, fw, fh)
        states = finger_states(pts)
        hands.append({
            "label": result.handedness[i][0].category_name,   # "Left" or "Right"
            "pts": pts,
            "states": states,
            "fist": is_fist(states),
        })

    # Guide graphics
    cv2.line(frame, (0, scroll_line_y), (fw, scroll_line_y), (255, 0, 255), 2)
    cv2.rectangle(frame, (frame_reduction, frame_reduction),
                  (fw - frame_reduction, fh - frame_reduction), (255, 0, 255), 2)

    # ----- Exit gesture: both hands are fists -----
    fist_count = sum(1 for h in hands if h["fist"])
    if len(hands) == 2 and fist_count == 2:
        if exit_start == 0:
            exit_start = now
        elapsed = now - exit_start
        remaining = max(0, math.ceil(exit_hold_seconds - elapsed))
        cv2.putText(frame, f"EXIT IN: {remaining}s", (fw // 2 - 100, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if elapsed >= exit_hold_seconds:
            print("Exiting...")
            break
    else:
        exit_start = 0

    # ----- Per-hand actions -----
    status = "Idle"
    left_moving_now = False
    right_fist_seen = False

    for h in hands:
        label, pts, st = h["label"], h["pts"], h["states"]
        draw_hand(frame, pts)

        idx_x, idx_y = int(pts[8][0]), int(pts[8][1])
        label_pos = (idx_x, max(idx_y - 40, 20))

        thumb, index_up, middle_up, ring_up, pinky_up = st

        # ===== LEFT HAND: move and scroll =====
        if label == "Left":
            # Scroll: index + middle up, ring + pinky folded
            if index_up and middle_up and not ring_up and not pinky_up:
                status = "Scroll"
                cv2.putText(frame, "SCROLL", label_pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                mid_y = pts[12][1]
                if mid_y < scroll_line_y - scroll_dead_zone:
                    pyautogui.scroll(scroll_speed)
                    cv2.putText(frame, "UP", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                elif mid_y > scroll_line_y + scroll_dead_zone:
                    pyautogui.scroll(-scroll_speed)
                    cv2.putText(frame, "DOWN", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Move: only the index finger up
            elif index_up and not middle_up and not ring_up and not pinky_up:
                status = "Move"
                left_moving_now = True

                # On the first moving frame, start from where the cursor really is
                if not moving:
                    prev_x, prev_y = pyautogui.position()

                target_x = float(np.interp(idx_x, (frame_reduction, fw - frame_reduction), (0, screen_w - 1)))
                target_y = float(np.interp(idx_y, (frame_reduction, fh - frame_reduction), (0, screen_h - 1)))

                gap = math.hypot(target_x - prev_x, target_y - prev_y)
                if gap < jitter_dist:
                    target_x, target_y = prev_x, prev_y

                speed_factor = min(1.0, gap / 150)
                smooth = max_smooth - speed_factor * (max_smooth - min_smooth)

                cur_x = prev_x + (target_x - prev_x) / smooth
                cur_y = prev_y + (target_y - prev_y) / smooth
                cur_x = min(max(cur_x, 0), screen_w - 1)
                cur_y = min(max(cur_y, 0), screen_h - 1)

                pyautogui.moveTo(cur_x, cur_y)
                prev_x, prev_y = cur_x, cur_y
                cv2.circle(frame, (idx_x, idx_y), 8, (0, 255, 255), cv2.FILLED)

        # ===== RIGHT HAND: clicks =====
        elif label == "Right":
            palm = dist(pts[0], pts[9])                 # wrist to middle knuckle
            pinch_gap = dist(pts[4], pts[8])
            pinching = palm > 0 and (pinch_gap / palm) < pinch_ratio

            # Right click: fist held briefly (not while doing the exit gesture)
            if h["fist"] and fist_count != 2:
                right_fist_seen = True
                status = "Right fist"
                if right_fist_since == 0:
                    right_fist_since = now
                if now - right_fist_since >= right_fist_hold:
                    cv2.putText(frame, "RIGHT CLICK", label_pos, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    if now - last_right_click > right_click_cooldown:
                        pyautogui.rightClick()
                        last_right_click = now
                        print("Right Click")

            # Left click: pinch
            elif pinching:
                status = "Pinch"
                since_right = now - last_right_click
                if since_right > safety_delay:
                    cv2.circle(frame, (idx_x, idx_y), 15, (0, 255, 0), cv2.FILLED)
                    if now - last_left_click > click_cooldown:
                        pyautogui.click()
                        last_left_click = now
                        print("Left Click")
                else:
                    wait = round(safety_delay - since_right, 1)
                    cv2.putText(frame, f"Wait {wait}s", label_pos, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if not right_fist_seen:
        right_fist_since = 0
    moving = left_moving_now

    # ----- HUD -----
    fps = 1.0 / max(now - prev_frame_time, 1e-6)
    prev_frame_time = now
    cv2.putText(frame, f"{status}  |  {int(fps)} FPS", (10, 25),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- Cleanup ----------
hand_detector.close()
cap.release()
cv2.destroyAllWindows()