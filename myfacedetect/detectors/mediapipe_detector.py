"""
MediaPipe Detector
Google's MediaPipe face detection with improved configuration.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List
from .base_detector import BaseDetector
from ..core import FaceDetectionResult
import logging

logger = logging.getLogger(__name__)


class MediaPipeDetector(BaseDetector):
    """MediaPipe face detector with enhanced configuration."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "auto",
                 model_selection: int = 0, min_detection_confidence: float = 0.5):
        super().__init__(confidence_threshold, device)
        self.model_selection = model_selection  # 0 or 1
        self.min_detection_confidence = min_detection_confidence
        self.face_detection = None
        self.load_model()

    def load_model(self):
        """Load MediaPipe face detection model."""
        try:
            # Initialize MediaPipe face detection
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_detection_confidence
            )
            logger.info(
                f"MediaPipe detector loaded (model_selection={self.model_selection})")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe model: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB for MediaPipe."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using MediaPipe."""
        if self.face_detection is None:
            self.load_model()

        try:
            # Convert to RGB
            rgb_image = self.preprocess_image(image)
            h, w = image.shape[:2]

            # Process image
            results = self.face_detection.process(rgb_image)
            faces = []

            if results.detections:
                for detection in results.detections:
                    # Extract bounding box
                    bbox = detection.location_data.relative_bounding_box

                    # Convert to absolute coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # Get confidence score
                    confidence = detection.score[0] if detection.score else 0.0

                    # Create result
                    result = FaceDetectionResult(
                        bbox=(x, y, width, height),
                        confidence=confidence,
                        method="mediapipe"
                    )
                    faces.append(result)

            return self.postprocess_results(faces)

        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return []
