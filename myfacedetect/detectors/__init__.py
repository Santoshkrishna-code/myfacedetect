"""
Detectors Module
Provides modular face detection capabilities.
"""
from .base_detector import BaseDetector
from .haar_detector import HaarDetector
from .mediapipe_detector import MediaPipeDetector
from .ensemble_detector import EnsembleDetector
from .detector_factory import DetectorFactory

# Try to import optional detectors
try:
    from .retinaface_detector import RetinaFaceDetector
    RETINAFACE_AVAILABLE = True
except ImportError:
    RetinaFaceDetector = None
    RETINAFACE_AVAILABLE = False

try:
    from .yolov8_detector import YOLOv8Detector
    YOLOV8_AVAILABLE = True
except ImportError:
    YOLOv8Detector = None
    YOLOV8_AVAILABLE = False

__all__ = [
    "BaseDetector",
    "HaarDetector",
    "MediaPipeDetector",
    "EnsembleDetector",
    "DetectorFactory"
]

# Add optional detectors to __all__ if available
if RETINAFACE_AVAILABLE:
    __all__.append("RetinaFaceDetector")
if YOLOV8_AVAILABLE:
    __all__.append("YOLOv8Detector")
