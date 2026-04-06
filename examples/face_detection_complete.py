#!/usr/bin/env python3
"""All-in-one CPU-only face detection utility with menu.

Usage:
  python examples/face_detection_complete.py
"""
import os
import cv2

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class FaceDetector:
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        if HAS_MEDIAPIPE:
            self.mp_fd = mp.solutions.face_detection
            self.mp_draw = mp.solutions.drawing_utils

    def detect_image(self, path):
        from detect_faces_image import detect_faces_in_image
        detect_faces_in_image(path, min_confidence=self.confidence)

    def detect_live(self):
        from detect_faces_live import run_live
        run_live(min_confidence=self.confidence)

    def detect_batch(self, folder):
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
        print(f'Processing {len(files)} images')
        for f in files:
            print('->', f)
            self.detect_image(os.path.join(folder, f))


def main():
    fd = FaceDetector(confidence=0.5)
    print('1) Detect image')
    print('2) Live webcam')
    print('3) Batch folder')
    choice = input('Select (1/2/3): ').strip()
    if choice == '1':
        p = input('Image path: ').strip()
        fd.detect_image(p)
    elif choice == '2':
        fd.detect_live()
    elif choice == '3':
        d = input('Folder path: ').strip()
        fd.detect_batch(d)
    else:
        print('Invalid option')


if __name__ == '__main__':
    main()
