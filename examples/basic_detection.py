#!/usr/bin/endef main():
"""Run basic face detection example."""
from myfacedetect import detect_faces
import numpy as np
import cv2
print("MyFaceDetect - Basic Detection Example")
print("=" * 40)thon3
"""
Basic Face Detection Example
===========================

This example demonstrates basic face detection using MyFaceDetect library.
"""


def main():
    """Run basic face detection example."""
    print("🔍 MyFaceDetect - Basic Detection Example")
    print("=" * 45)

    # Create a sample test image
    test_image = np.ones((300, 300, 3), dtype=np.uint8) * 128

    # Draw a simple face-like pattern
    cv2.ellipse(test_image, (150, 150), (80, 100),
                0, 0, 360, (200, 180, 160), -1)
    cv2.circle(test_image, (125, 130), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(test_image, (175, 130), 8, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(test_image, (150, 170), (20, 10), 0,
                0, 180, (100, 80, 80), 2)  # Mouth

    # Detect faces using different methods
    methods = ['haar', 'mediapipe']

    for method in methods:
        print(f"\n🔍 Testing {method.upper()} detection...")
        try:
            faces = detect_faces(test_image, method=method)
            print(f"✅ Found {len(faces)} faces with {method}")

            for i, face in enumerate(faces):
                x, y, w, h = face.bbox
                confidence = face.confidence
                print(
                    f"   Face {i+1}: ({x}, {y}, {w}, {h}) - Confidence: {confidence:.3f}")

        except Exception as e:
            print(f"❌ {method} detection failed: {e}")

    print("\n✅ Basic detection example completed!")


if __name__ == "__main__":
    main()
