"""
MyFaceDetect v0.3.0 - Advanced Face Detection & Recognition Library
===================================================================

A production-ready, CPU-optimized Python library for face detection,
recognition, and analysis with comprehensive features and robust architecture.

🚀 **Key Features:**
- Multiple detection methods (Haar, MediaPipe, YOLOv8, RetinaFace, Ensemble)
- Advanced recognition with OpenCV LBPH and ArcFace embeddings
- Real-time processing with webcam integration
- Interactive training system with sample capture
- CPU-only execution mode for broad compatibility
- Comprehensive augmentation testing and validation
- Privacy protection and security features
- Intelligent caching and performance optimization

🔧 **Quick Start:**

**Basic Detection:**
    from myfacedetect import detect_faces
    faces = detect_faces("image.jpg", method="mediapipe")

**Real-time Processing:**
    from myfacedetect import detect_faces_realtime
    detect_faces_realtime(method="haar")

**Modern API:**
    from myfacedetect.detectors import DetectorFactory
    from myfacedetect.recognition import FaceRecognizer
    
    detector = DetectorFactory.create_detector('ensemble')
    recognizer = FaceRecognizer('opencv')
    
    faces = detector.detect(image)
    result, similarity = recognizer.recognize(face_image)

**Testing & Validation:**
    from myfacedetect.testing import FaceAugmentationTester, FaceLabelingSystem
    
    tester = FaceAugmentationTester(recognizer, detector)
    results = tester.test_recognition_robustness(image, "PersonName")
"""

# Version information
from .__version__ import (
    __version__, __title__, __author__, __email__, __license__,
    __description__, __url__, VERSION_INFO, BUILD_STATUS, RELEASE_DATE
)

# Core functionality
from .core import (
    detect_faces,
    detect_faces_realtime,
    FaceDetectionResult,
    batch_detect_faces,
    save_face_crops,
    FaceDetector
)

# Modern modular architecture
from .config_manager import ConfigManager
from .detectors import DetectorFactory

# Essential components only (production-ready)
__all__ = [
    # Version info
    '__version__',
    '__title__',
    '__author__',
    '__email__',
    '__license__',
    '__description__',
    '__url__',
    'VERSION_INFO',
    'BUILD_STATUS',
    'RELEASE_DATE',

    # Core API
    'detect_faces',
    'detect_faces_realtime',
    'FaceDetectionResult',
    'batch_detect_faces',
    'save_face_crops',
    'FaceDetector',

    # Modern architecture
    'ConfigManager',
    'DetectorFactory'
]
