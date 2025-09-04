"""
Haar Cascade Detector
Traditional OpenCV Haar cascade face detection.
"""
import cv2
import numpy as np
from typing import List
from .base_detector import BaseDetector
from ..core import FaceDetectionResult
import logging

logger = logging.getLogger(__name__)


class HaarDetector(BaseDetector):
    """Haar Cascade face detector."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "cpu",
                 scale_factor: float = 1.05, min_neighbors: int = 3,
                 min_size: tuple = (20, 20)):
        super().__init__(confidence_threshold, device)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascades = {}
        self.load_model()

    def load_model(self):
        """Load Haar cascade models."""
        try:
            # Load multiple cascade classifiers for better detection
            cascade_files = {
                'frontal_default': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                'frontal_alt': cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
                'frontal_alt2': cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
                'profile': cv2.data.haarcascades + 'haarcascade_profileface.xml'
            }

            for name, path in cascade_files.items():
                try:
                    self.cascades[name] = cv2.CascadeClassifier(path)
                    if self.cascades[name].empty():
                        logger.warning(f"Failed to load {name} cascade")
                    else:
                        logger.debug(f"Loaded {name} cascade successfully")
                except Exception as e:
                    logger.warning(f"Error loading {name} cascade: {e}")

            if not self.cascades:
                raise RuntimeError("No Haar cascades could be loaded")

            logger.info(
                f"HaarDetector loaded with {len(self.cascades)} cascades")

        except Exception as e:
            logger.error(f"Failed to load Haar cascades: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply histogram equalization."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better detection
        return cv2.equalizeHist(gray)

    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using Haar cascades."""
        gray = self.preprocess_image(image)
        all_faces = []

        # Use multiple cascades for better detection
        for cascade_name, cascade in self.cascades.items():
            if cascade.empty():
                continue

            try:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # Convert to FaceDetectionResult objects
                for (x, y, w, h) in faces:
                    # Haar cascades don't provide confidence, so use a fixed value
                    confidence = 0.8  # Reasonable default
                    result = FaceDetectionResult(
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        method=f"haar_{cascade_name}"
                    )
                    all_faces.append(result)

            except Exception as e:
                logger.warning(f"Error in {cascade_name} detection: {e}")

        # Remove duplicates using Non-Maximum Suppression
        filtered_faces = self._apply_nms(all_faces, overlap_threshold=0.3)
        return self.postprocess_results(filtered_faces)

    def _apply_nms(self, faces: List[FaceDetectionResult], overlap_threshold: float = 0.3) -> List[FaceDetectionResult]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(faces) <= 1:
            return faces

        # Convert to format suitable for cv2.dnn.NMSBoxes
        boxes = []
        confidences = []

        for face in faces:
            boxes.append([face.x, face.y, face.width, face.height])
            confidences.append(float(face.confidence))

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            score_threshold=0.0,  # We already filtered by confidence
            nms_threshold=overlap_threshold
        )

        # Return filtered faces
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i] for i in indices]
        else:
            return []
