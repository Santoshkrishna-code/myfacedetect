#!/usr/bin/env python3
"""
Face Recognition Example
=======================

This example demonstrates face recognition with training and identification.
"""

import cv2
import numpy as np
from myfacedetect.detectors import DetectorFactory
from myfacedetect.recognition import FaceRecognizer


def create_sample_face(person_id, variation=0):
    """Create a sample face for demonstration."""
    # Create base face
    face = np.ones((150, 150, 3), dtype=np.uint8) * (120 + variation * 10)

    # Face shape varies by person
    if person_id == 1:
        cv2.ellipse(face, (75, 75), (50, 60), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(face, (60, 65), 6, (50, 50, 50), -1)
        cv2.circle(face, (90, 65), 6, (50, 50, 50), -1)
        cv2.ellipse(face, (75, 95), (15, 8), 0, 0, 180, (100, 80, 80), 2)
    elif person_id == 2:
        cv2.ellipse(face, (75, 75), (55, 65), 0, 0, 360, (190, 170, 150), -1)
        cv2.circle(face, (58, 68), 7, (40, 40, 40), -1)
        cv2.circle(face, (92, 68), 7, (40, 40, 40), -1)
        cv2.ellipse(face, (75, 100), (18, 10), 0, 0, 180, (90, 70, 70), 2)
    else:
        cv2.ellipse(face, (75, 75), (45, 55), 0, 0, 360, (210, 190, 170), -1)
        cv2.circle(face, (62, 70), 5, (60, 60, 60), -1)
        cv2.circle(face, (88, 70), 5, (60, 60, 60), -1)
        cv2.ellipse(face, (75, 90), (12, 6), 0, 0, 180, (110, 90, 90), 2)

    return face


def main():
    """Run face recognition example."""
    print("🧠 MyFaceDetect - Face Recognition Example")
    print("=" * 47)

    # Initialize systems
    detector = DetectorFactory.create_detector('haar')
    recognizer = FaceRecognizer('opencv')

    print("✅ Detection and recognition systems initialized")

    # Create training samples
    print("\n📚 Creating training samples...")
    people = ["Alice", "Bob", "Charlie"]

    for i, person in enumerate(people, 1):
        print(f"   Training {person}...")

        # Create multiple samples for each person
        for variation in range(3):
            sample_face = create_sample_face(i, variation)

            # Add to recognizer
            try:
                recognizer.add_face(sample_face, person)
                print(f"     ✅ Added sample {variation + 1} for {person}")
            except Exception as e:
                print(f"     ❌ Failed to add sample for {person}: {e}")

    print(
        f"\n🧠 Training completed. Database contains: {recognizer.get_all_names()}")

    # Test recognition
    print("\n🔍 Testing recognition...")

    for i, person in enumerate(people, 1):
        print(f"\n   Testing {person}:")

        # Create test sample (slightly different from training)
        test_face = create_sample_face(i, variation=5)  # Different variation

        try:
            # Recognize face
            result, similarity = recognizer.recognize(test_face)

            if result:
                status = "✅ RECOGNIZED" if result == person else "❓ MISIDENTIFIED"
                print(
                    f"     {status}: Predicted={result}, Similarity={similarity:.3f}")
            else:
                print(f"     ❌ NOT RECOGNIZED: Similarity={similarity:.3f}")

        except Exception as e:
            print(f"     ❌ Recognition failed: {e}")

    # Test with unknown person
    print(f"\n   Testing unknown person:")
    unknown_face = create_sample_face(99, variation=0)  # Completely different

    try:
        result, similarity = recognizer.recognize(unknown_face)
        if result:
            print(
                f"     ❓ UNEXPECTED: Predicted={result}, Similarity={similarity:.3f}")
        else:
            print(f"     ✅ CORRECTLY UNKNOWN: Similarity={similarity:.3f}")
    except Exception as e:
        print(f"     ❌ Recognition failed: {e}")

    print("\n✅ Face recognition example completed!")


if __name__ == "__main__":
    main()
