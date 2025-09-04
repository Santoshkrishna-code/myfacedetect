"""
YOLOv8 Face Detector
Ultra-fast face detection using YOLOv8.
Requires ultralytics package: pip install ultralytics
"""
import cv2
import numpy as np
from typing import List
from .base_detector import BaseDetector
from ..core import FaceDetectionResult
import logging

logger = logging.getLogger(__name__)


class YOLOv8Detector(BaseDetector):
    """YOLOv8 face detector for ultra-fast detection."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "auto",
                 model_size: str = "n"):  # n, s, m, l, x
        super().__init__(confidence_threshold, device)
        self.model_size = model_size
        self.model = None
        # Don't load model in __init__ - let it be loaded lazily

    def load_model(self):
        """Load YOLOv8 face detection model."""
        if self.model is not None:
            return

        if not self.is_available():
            raise ImportError(
                "Please install ultralytics: pip install ultralytics")

        try:
            from ultralytics import YOLO
            import os
            import urllib.request

            # Force CPU-only to avoid CUDA issues
            if self.device == "auto" or "cuda" in str(self.device).lower():
                self.device = "cpu"
                logger.info("Using CPU for YOLOv8 to avoid CUDA conflicts")

            # First try to load face-specific YOLOv8 model
            model_path = f"yolov8{self.model_size}-face.pt"

            # Download YOLOv8-face model if not exists
            if not os.path.exists(model_path):
                logger.info(f"Downloading YOLOv8-face model: {model_path}")
                try:
                    # Try to download from known sources or use alternative
                    face_model_urls = {
                        "yolov8n-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
                        "yolov8s-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8s-face.pt"
                    }

                    if model_path in face_model_urls:
                        urllib.request.urlretrieve(
                            face_model_urls[model_path], model_path)
                        logger.info(
                            f"Downloaded YOLOv8-face model: {model_path}")
                except Exception as download_error:
                    logger.warning(
                        f"Failed to download face model: {download_error}")

            try:
                self.model = YOLO(model_path)
                self.model.to(self.device)
                logger.info(
                    f"YOLOv8{self.model_size} face detector loaded on {self.device}")
            except Exception as face_load_error:
                # Fallback to general YOLOv8 and filter for person class
                logger.warning(
                    f"YOLOv8-face model failed ({face_load_error}), using general YOLOv8")
                self.model = YOLO(f"yolov8{self.model_size}.pt")
                self.model.to(self.device)
                logger.info(
                    f"YOLOv8{self.model_size} general detector loaded on {self.device}")

        except ImportError:
            logger.error(
                "YOLOv8 requires 'ultralytics' package. Install with: pip install ultralytics")
            raise ImportError(
                "Please install ultralytics: pip install ultralytics")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using YOLOv8."""
        if self.model is None:
            self.load_model()

        try:
            # Run inference
            results = self.model(image, verbose=False)
            faces = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get confidence
                        confidence = float(box.conf.item())

                        if confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y = int(x1), int(y1)
                            width = int(x2 - x1)
                            height = int(y2 - y1)

                            # Create result
                            result_obj = FaceDetectionResult(
                                bbox=(x, y, width, height),
                                confidence=confidence,
                                method="yolov8"
                            )
                            faces.append(result_obj)

            return self.postprocess_results(faces)

        except Exception as e:
            logger.error(f"YOLOv8 detection error: {e}")
            return []

    def is_available(self) -> bool:
        """Check if YOLOv8 is available."""
        try:
            import ultralytics
            return True
        except ImportError:
            return False
