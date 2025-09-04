"""
RetinaFace Detector
State-of-the-art face detection using RetinaFace.
Requires insightface package: pip install insightface
"""
import cv2
import numpy as np
from typing import List, Optional
from .base_detector import BaseDetector
from ..core import FaceDetectionResult
import logging

logger = logging.getLogger(__name__)


class RetinaFaceDetector(BaseDetector):
    """RetinaFace detector for high accuracy face detection."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "auto"):
        super().__init__(confidence_threshold, device)
        self.model = None
        # Don't load model in __init__ - let it be loaded lazily

    def load_model(self):
        """Load RetinaFace model."""
        if self.model is not None:
            return

        if not self.is_available():
            raise ImportError(
                "Please install insightface: pip install insightface")

        try:
            # Try to import insightface
            import insightface
            from insightface.app import FaceAnalysis

            # Initialize model
            self.model = FaceAnalysis(
                name='buffalo_l', allowed_modules=['detection'])

            # Always use CPU to avoid CUDA issues
            ctx_id = -1  # Force CPU

            self.model.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info(f"RetinaFace detector loaded on CPU")

        except ImportError:
            logger.error(
                "RetinaFace requires 'insightface' package. Install with: pip install insightface")
            raise ImportError(
                "Please install insightface: pip install insightface")
        except Exception as e:
            logger.error(f"Failed to load RetinaFace model: {e}")
            raise

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import onnxruntime as ort
            return 'CUDAExecutionProvider' in ort.get_available_providers()
        except ImportError:
            return False

    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using RetinaFace."""
        if self.model is None:
            self.load_model()

        try:
            # RetinaFace expects RGB format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get detections
            faces_data = self.model.get(rgb_image)
            faces = []

            for face in faces_data:
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                width = x2 - x
                height = y2 - y

                # Get confidence (detection score)
                confidence = float(face.det_score)

                # Create result
                result = FaceDetectionResult(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    method="retinaface"
                )

                # Add landmarks if available
                if hasattr(face, 'landmark_2d_106'):
                    result.landmarks = face.landmark_2d_106
                elif hasattr(face, 'kps'):
                    result.landmarks = face.kps

                faces.append(result)

            return self.postprocess_results(faces)

        except Exception as e:
            logger.error(f"RetinaFace detection error: {e}")
            return []

    def is_available(self) -> bool:
        """Check if RetinaFace is available."""
        try:
            import insightface
            return True
        except ImportError:
            return False
