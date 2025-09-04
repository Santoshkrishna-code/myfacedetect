#!/usr/bin/env python3
"""
MyFaceDetect CLI Interface
=========================

Command-line interface for MyFaceDetect library.
"""

import argparse
import sys
import cv2
from pathlib import Path

try:
    from .detectors.detector_factory import DetectorFactory
    from .recognition.face_recognition import FaceRecognizer
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from myfacedetect.detectors.detector_factory import DetectorFactory
    from myfacedetect.recognition.face_recognition import FaceRecognizer


def detect_faces(image_path, method='mediapipe', output=None):
    """Detect faces in an image."""
    print(f"🔍 Detecting faces in: {image_path}")
    print(f"📊 Using method: {method}")

    try:
        detector = DetectorFactory.create_detector(method)
        faces = detector.detect(cv2.imread(image_path))

        print(f"✅ Found {len(faces)} face(s)")

        for i, face in enumerate(faces, 1):
            bbox = face.bbox
            print(f"  Face {i}: x={bbox[0]}, y={bbox[1]}, "
                  f"w={bbox[2]}, h={bbox[3]}, "
                  f"confidence={face.confidence:.3f}")

        if output:
            # Save annotated image
            image = cv2.imread(image_path)
            for face in faces:
                x, y, w, h = face.bbox
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f'{face.confidence:.2f}', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite(output, image)
            print(f"💾 Saved annotated image to: {output}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def recognize_faces(image_path, database_path=None):
    """Recognize faces in an image."""
    print(f"👤 Recognizing faces in: {image_path}")

    try:
        recognizer = FaceRecognizer()

        if database_path and Path(database_path).exists():
            recognizer.load_database(database_path)
            print(f"📚 Loaded database: {database_path}")

        detector = DetectorFactory.create_detector('mediapipe')
        image = cv2.imread(image_path)
        faces = detector.detect(image)

        print(f"🔍 Found {len(faces)} face(s)")

        for i, face in enumerate(faces, 1):
            # Extract face for recognition
            x, y, w, h = face.bbox
            face_roi = image[y:y+h, x:x+w]

            try:
                result, similarity = recognizer.recognize(face_roi)

                if result != 'unknown':
                    print(f"  Face {i}: {result} "
                          f"(similarity: {similarity:.3f})")
                else:
                    print(f"  Face {i}: Unknown person")
            except Exception as e:
                print(f"  Face {i}: Recognition error - {str(e)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MyFaceDetect - Face Detection and Recognition CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detect image.jpg                    # Detect faces
  %(prog)s detect image.jpg -m haar            # Use Haar cascades
  %(prog)s detect image.jpg -o output.jpg      # Save annotated image
  %(prog)s recognize image.jpg                 # Recognize faces
  %(prog)s recognize image.jpg -d faces.pkl    # Use custom database
        """
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Detection command
    detect_parser = subparsers.add_parser(
        'detect', help='Detect faces in image')
    detect_parser.add_argument('image', help='Path to input image')
    detect_parser.add_argument('-m', '--method',
                               choices=['mediapipe', 'haar', 'yolov8'],
                               default='mediapipe',
                               help='Detection method (default: mediapipe)')
    detect_parser.add_argument('-o', '--output',
                               help='Save annotated image to this path')

    # Recognition command
    recognize_parser = subparsers.add_parser(
        'recognize', help='Recognize faces in image')
    recognize_parser.add_argument('image', help='Path to input image')
    recognize_parser.add_argument('-d', '--database',
                                  help='Path to face database file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print("🚀 MyFaceDetect CLI v0.3.0")
    print("-" * 30)

    if args.command == 'detect':
        detect_faces(args.image, args.method, args.output)
    elif args.command == 'recognize':
        recognize_faces(args.image, args.database)


if __name__ == "__main__":
    main()
