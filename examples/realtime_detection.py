#!/usr/bin/env python3
"""
Real-time Face Detection Example
===============================

This example demonstrates real-time face detection using webcam.
"""

import cv2
from myfacedetect.detectors import DetectorFactory


def main():
    """Run real-time face detection example."""
    print("📹 MyFaceDetect - Real-time Detection Example")
    print("=" * 48)

    # Initialize detector
    detector = DetectorFactory.create_detector('haar')
    print("✅ Face detector initialized")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("\n📹 Starting real-time detection...")
    print("Press 'q' to quit, 's' to save screenshot")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect faces every few frames for performance
            if frame_count % 3 == 0:
                faces = detector.detect(frame)

                # Draw bounding boxes
                for face in faces:
                    x, y, w, h = face.bbox
                    confidence = face.confidence

                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

                    # Draw confidence
                    label = f"Face ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Real-time Face Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'detection_frame_{frame_count}.jpg', frame)
                print(f"Screenshot saved: detection_frame_{frame_count}.jpg")

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Real-time detection example completed!")


if __name__ == "__main__":
    main()
