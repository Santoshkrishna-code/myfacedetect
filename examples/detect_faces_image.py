#!/usr/bin/env python3
"""Detect faces in a static image (CPU-only, uses MediaPipe if available).

Usage:
  python examples/detect_faces_image.py path/to/image.jpg

Saves output to image_detected.<ext>
"""
import sys
import cv2

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


def detect_faces_in_image(image_path, min_confidence=0.5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image '{image_path}'")
        return

    ih, iw = image.shape[:2]

    if HAS_MEDIAPIPE:
        mp_face = mp.solutions.face_detection
        mp_draw = mp.solutions.drawing_utils
        with mp_face.FaceDetection(min_detection_confidence=min_confidence) as fd:
            results = fd.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.detections:
                print("No faces detected")
            else:
                print(f"Detected {len(results.detections)} face(s)")
                for det in results.detections:
                    mp_draw.draw_detection(image, det)
    else:
        # Fallback to OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            print("No faces detected (Haar cascade)")
        else:
            print(f"Detected {len(faces)} face(s) (Haar cascade)")
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save result
    out_path = image_path.rsplit('.', 1)
    if len(out_path) == 2:
        out = out_path[0] + '_detected.' + out_path[1]
    else:
        out = image_path + '_detected.jpg'

    cv2.imwrite(out, image)
    print(f"Saved detection result to: {out}")
    cv2.imshow('Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python examples/detect_faces_image.py path/to/image.jpg')
        sys.exit(1)
    detect_faces_in_image(sys.argv[1])
