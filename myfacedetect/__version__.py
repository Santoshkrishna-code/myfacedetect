"""
MyFaceDetect Version Information
"""

__title__ = "myfacedetect"
__version__ = "0.3.0"
__author__ = "Santoshkrishna"
__email__ = "santoshkrishna@example.com"
__license__ = "MIT"
__description__ = "Advanced Face Detection and Recognition Library"
__url__ = "https://github.com/Santoshkrishna-code/myfacedetect"

# Version components
VERSION_MAJOR = 0
VERSION_MINOR = 3
VERSION_PATCH = 0
VERSION_INFO = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

# Build info
BUILD_STATUS = "stable"
RELEASE_DATE = "2025-09-04"

# Features
SUPPORTED_DETECTORS = [
    "Haar Cascade",
    "MediaPipe",
    "Ensemble (Haar + MediaPipe)",
    "YOLOv8 (optional)",
    "RetinaFace (optional)"
]

SUPPORTED_RECOGNIZERS = [
    "OpenCV LBPH",
    "ArcFace (optional)"
]

# System requirements
MIN_PYTHON_VERSION = "3.8"
REQUIRED_PACKAGES = [
    "opencv-python>=4.5.0",
    "mediapipe>=0.8.0",
    "numpy>=1.19.0",
    "Pillow>=8.0.0",
    "pyyaml>=5.4.0"
]

OPTIONAL_PACKAGES = [
    "ultralytics>=8.0.0",  # For YOLOv8
    "insightface>=0.7.0",  # For ArcFace
    "onnxruntime>=1.12.0"  # For ONNX models
]
