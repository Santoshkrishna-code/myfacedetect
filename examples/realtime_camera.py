#!/usr/bin/env python3
"""Live face detection from the default camera.

Usage:
  python examples/realtime_camera.py         # open camera 0
  python examples/realtime_camera.py --cam 1  # open camera 1

Press 'q' to quit, 's' to save a snapshot.

This script will try to use MediaPipe's face detection if installed,
otherwise it falls back to OpenCV's Haar cascade detector.
"""
import time
import argparse
import os

try:
    import cv2
except Exception as e:
    raise SystemExit("OpenCV is required: pip install opencv-python\n" + str(e))

HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


def draw_bbox(frame, x1, y1, x2, y2, color=(0, 255, 0), label=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_camera(cam_index=0, width=640, height=480, min_confidence=0.5):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}")
        return

    if HAS_MEDIAPIPE:
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(min_detection_confidence=min_confidence)
        use_mediapipe = True
        print("Using MediaPipe face detection")
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        use_mediapipe = False
        print("MediaPipe not found — falling back to OpenCV Haar cascade")

    prev = time.time()
    fps = 0.0
    frame_count = 0

    cv2.namedWindow("Live Face Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        h, w = frame.shape[:2]

        if use_mediapipe:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = x1 + int(bbox.width * w)
                    y2 = y1 + int(bbox.height * h)
                    score = det.score[0] if det.score else None
                    label = f"{score:.2f}" if score is not None else None
                    draw_bbox(frame, x1, y1, x2, y2, label=label)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                draw_bbox(frame, x, y, x + fw, y + fh)

        # FPS
        frame_count += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = frame_count / (now - prev)
            prev = now
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Live Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved snapshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Live face detection from camera")
    p.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--width", type=int, default=640, help="Frame width")
    p.add_argument("--height", type=int, default=480, help="Frame height")
    p.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence for MediaPipe detector")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_camera(cam_index=args.cam, width=args.width, height=args.height, min_confidence=args.min_confidence)
    except KeyboardInterrupt:
        print("Interrupted by user")
