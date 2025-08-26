"""
MyFaceDetect - A comprehensive face detection library
====================================================

A Python library for face detection in images and real-time video streams
using OpenCV Haar cascades and MediaPipe.

Main Functions:
- detect_faces: Detect faces in static images
- detect_faces_realtime: Real-time face detection via webcam
- batch_detect_faces: Process multiple images
- save_face_crops: Extract and save individual faces

Example Usage:
    from myfacedetect import detect_faces, detect_faces_realtime
    
    # Static image detection
    faces = detect_faces("image.jpg", method="mediapipe")
    print(f"Found {len(faces)} faces")
    
    # Real-time detection
    detect_faces_realtime(method="both")
"""

from .core import (
    detect_faces,
    detect_faces_realtime,
    FaceDetectionResult,
    batch_detect_faces,
    save_face_crops
)

__version__ = "0.2.2"
__author__ = "B Santosh Krishna"
__email__ = "santoshkrishnabandla@gmail.com"
__description__ = "Enhanced face detection library with multiple methods"

# Make main functions available at package level
__all__ = [
    'detect_faces',
    'detect_faces_realtime',
    'FaceDetectionResult',
    'batch_detect_faces',
    'save_face_crops'
]
