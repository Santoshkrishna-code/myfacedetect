# CPU-only Face Detection Setup Guide

This guide explains how to set up and run the provided CPU-friendly face detection scripts in this repository.

Prerequisites
- Python 3.8+
- Webcam for live detection (optional)
- 4GB RAM minimum (8GB recommended)

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv facedetect_env
facedetect_env\Scripts\activate
```

2) Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3) Scripts included:
- `examples/detect_faces_image.py` — run on single images
- `examples/detect_faces_live.py` — live webcam detection (press `q` to quit, `s` to save snapshot)
- `examples/face_detection_complete.py` — menu-driven utility
- `scripts/test_setup.py` — verifies OpenCV/MediaPipe and webcam

4) Run quick test:

```powershell
python scripts/test_setup.py
```

5) Run live detection:

```powershell
python examples/detect_faces_live.py
```

6) Troubleshooting & Tips
- If webcam not found, try different camera index or check privacy settings
- Lower resolution (640x480) improves CPU FPS
- If `mediapipe` fails to install, the scripts will fallback to OpenCV Haar cascade

7) Next steps
- Add face recognition (requires dataset and model)
- Add a Dockerfile if you want reproducible environments

