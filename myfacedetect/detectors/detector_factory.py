"""
Detector Factory
Central registry for all face detectors.
"""
from typing import Dict, Type, Optional
from .base_detector import BaseDetector
from .haar_detector import HaarDetector
from .mediapipe_detector import MediaPipeDetector
from .ensemble_detector import EnsembleDetector
import logging

logger = logging.getLogger(__name__)


class DetectorFactory:
    """Factory for creating face detectors."""

    _detectors: Dict[str, Type[BaseDetector]] = {
        "haar": HaarDetector,
        "mediapipe": MediaPipeDetector,
        "ensemble": EnsembleDetector,
    }

    @classmethod
    def register_detector(cls, name: str, detector_class: Type[BaseDetector]):
        """Register a new detector."""
        cls._detectors[name] = detector_class
        logger.info(f"Registered detector: {name}")

    @classmethod
    def create_detector(cls, name: str, config: Optional[Dict] = None, **kwargs) -> BaseDetector:
        """Create a detector instance."""
        if name not in cls._detectors:
            available = list(cls._detectors.keys())
            raise ValueError(
                f"Unknown detector '{name}'. Available: {available}")

        detector_class = cls._detectors[name]

        # Extract relevant config parameters for the detector
        detector_kwargs = {}
        if config:
            # Convert ConfigManager to dict if needed
            if hasattr(config, 'config'):
                config_dict = config.config
            elif hasattr(config, '__dict__'):
                config_dict = config.__dict__
            elif isinstance(config, dict):
                config_dict = config
            else:
                config_dict = {}

            # Common parameters that all detectors should accept
            common_params = ['confidence_threshold', 'device']

            # Detector-specific parameter mapping
            if name == 'haar':
                specific_params = ['scale_factor', 'min_neighbors', 'min_size']
            elif name == 'mediapipe':
                specific_params = ['model_selection',
                                   'min_detection_confidence']
            elif name == 'ensemble':
                specific_params = ['detectors',
                                   'voting_threshold', 'nms_threshold']
            else:
                specific_params = []

            # Copy relevant parameters
            for param in common_params + specific_params:
                if param in config_dict:
                    detector_kwargs[param] = config_dict[param]

        # Merge with any additional kwargs
        detector_kwargs.update(kwargs)

        return detector_class(**detector_kwargs)

    @classmethod
    def list_detectors(cls) -> list:
        """List available detectors."""
        return list(cls._detectors.keys())

    @classmethod
    def get_available_detectors(cls) -> list:
        """Get list of available detectors."""
        return cls.list_detectors()

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if detector is available."""
        if name not in cls._detectors:
            return False

        # For optional detectors, check if they can be loaded
        if name in ["retinaface", "yolov8"]:
            try:
                detector = cls._detectors[name]()
                return detector.is_available()
            except:
                return False

        return True

# Register optional detectors if available


def _register_optional_detectors():
    """Register optional detectors if their dependencies are available."""

    # Try to register RetinaFace
    try:
        from .retinaface_detector import RetinaFaceDetector
        if RetinaFaceDetector().is_available():
            DetectorFactory.register_detector("retinaface", RetinaFaceDetector)
    except ImportError:
        logger.debug("RetinaFace detector not available (missing insightface)")
    except Exception as e:
        logger.debug(f"RetinaFace detector not available: {e}")

    # Try to register YOLOv8
    try:
        from .yolov8_detector import YOLOv8Detector
        if YOLOv8Detector().is_available():
            DetectorFactory.register_detector("yolov8", YOLOv8Detector)
    except ImportError:
        logger.debug("YOLOv8 detector not available (missing ultralytics)")
    except Exception as e:
        logger.debug(f"YOLOv8 detector not available: {e}")


# Register optional detectors on import
_register_optional_detectors()
