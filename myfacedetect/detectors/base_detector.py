"""
Base Detector Interface
All face detectors must inherit from this base class.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import numpy as np
from ..core import FaceDetectionResult


class BaseDetector(ABC):
    """Abstract base class for face detectors."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "auto"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.name = self.__class__.__name__.lower().replace('detector', '')

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces in image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of FaceDetectionResult objects
        """
        pass

    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Alias for detect method for backward compatibility.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of FaceDetectionResult objects
        """
        return self.detect(image)

    @abstractmethod
    def load_model(self):
        """Load detector model."""
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before detection."""
        return image

    def postprocess_results(self, results: List[FaceDetectionResult]) -> List[FaceDetectionResult]:
        """Postprocess detection results."""
        # Filter by confidence threshold
        return [r for r in results if r.confidence >= self.confidence_threshold]

    def __str__(self):
        return f"{self.name.capitalize()}Detector(confidence={self.confidence_threshold})"
