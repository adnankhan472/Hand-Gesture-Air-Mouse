# Hand Gesture Air Mouse

Control your computer's mouse with hand gestures using only a webcam.
Built with Python, OpenCV, MediaPipe and PyAutoGUI.

Demo video: [add your YouTube link here]

## Gestures

| Hand | Gesture | Action |
|---|---|---|
| Left | Index finger up | Move cursor |
| Left | Index + middle fingers up | Scroll (above the purple line = up, below = down) |
| Right | Pinch thumb + index | Left click |
| Right | Fist | Right click |
| Both | Fists for 5 seconds | Exit |

## Install and run

```bash
pip install -r requirements.txt
python AirMouse.py
```

The first run downloads the MediaPipe hand model (`hand_landmarker.task`) automatically.

## How it works

1. OpenCV captures webcam frames.
2. MediaPipe's HandLandmarker finds 21 points on each hand.
3. Python checks which fingers are extended and measures the thumb-index gap.
4. PyAutoGUI moves the real mouse, clicks and scrolls.

## Limitations

Needs good lighting, and it is not suited to precise tasks like drawing.

