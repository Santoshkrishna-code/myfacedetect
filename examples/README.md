# MyFaceDetect Examples

This directory contains example scripts demonstrating various features of the MyFaceDetect library.

## Available Examples

### 1. Basic Detection (`basic_detection.py`)
Demonstrates basic face detection using different methods:
- Haar Cascade detection
- MediaPipe detection
- Results comparison

**Usage:**
```bash
python examples/basic_detection.py
```

### 2. Real-time Detection (`realtime_detection.py`)
Shows real-time face detection using webcam:
- Live video processing
- Bounding box visualization
- Performance optimization

**Usage:**
```bash
python examples/realtime_detection.py
```

### 3. Face Recognition (`face_recognition.py`)
Demonstrates face recognition capabilities:
- Training with sample faces
- Recognition testing
- Unknown person detection

**Usage:**
```bash
python examples/face_recognition.py
```

## Requirements

Make sure you have MyFaceDetect installed and a webcam connected for real-time examples:

```bash
pip install -r requirements.txt
```

## Notes

- Examples use synthetic faces for demonstration purposes
- For production use, replace with real face images
- Webcam examples require camera permissions
- Press 'q' to quit real-time examples
