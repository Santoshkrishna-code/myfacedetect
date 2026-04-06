#!/usr/bin/env python3
"""Live webcam face detection (CPU-only).

Usage:
  python examples/detect_faces_live.py

Keys:
  q - quit
  s - save snapshot
"""
import time
import cv2

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


def run_live(cam_index=0, width=640, height=480, min_confidence=0.5):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}")
        return

    if HAS_MEDIAPIPE:
        mp_face = mp.solutions.face_detection
        mp_draw = mp.solutions.drawing_utils
        detector = mp_face.FaceDetection(min_detection_confidence=min_confidence)
        use_mp = True
        print('Using MediaPipe face detection')
    else:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(cascade_path)
        use_mp = False
        print('Using OpenCV Haar cascade')

    prev = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read frame')
            break

        h, w = frame.shape[:2]

        if use_mp:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if results.detections:
                for d in results.detections:
                    mp_draw.draw_detection(frame, d)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        frames += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = frames / (now - prev)
            prev = now
            frames = 0
        else:
            fps = 0.0

        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Live Face Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fn = f'snapshot_{int(time.time())}.jpg'
            cv2.imwrite(fn, frame)
            print(f'Saved {fn}')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_live()
