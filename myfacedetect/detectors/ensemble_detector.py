"""
Ensemble Detector
Combines multiple detectors for improved accuracy.
"""
import numpy as np
from typing import List, Dict
from .base_detector import BaseDetector
from .haar_detector import HaarDetector
from .mediapipe_detector import MediaPipeDetector
from ..core import FaceDetectionResult
import logging

logger = logging.getLogger(__name__)


class EnsembleDetector(BaseDetector):
    """Ensemble detector that combines multiple detection methods."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "auto",
                 detectors: List[str] = None, voting_threshold: int = 2):
        super().__init__(confidence_threshold, device)
        self.detectors = {}
        # Minimum votes for a face to be accepted
        self.voting_threshold = voting_threshold

        # Default detectors if none specified
        if detectors is None:
            detectors = ["haar", "mediapipe"]

        self.load_detectors(detectors)

    def load_detectors(self, detector_names: List[str]):
        """Load specified detectors."""
        detector_classes = {
            "haar": HaarDetector,
            "mediapipe": MediaPipeDetector,
        }

        # Add optional detectors if available
        try:
            from .retinaface_detector import RetinaFaceDetector
            if RetinaFaceDetector().is_available():
                detector_classes["retinaface"] = RetinaFaceDetector
        except:
            pass

        try:
            from .yolov8_detector import YOLOv8Detector
            if YOLOv8Detector().is_available():
                detector_classes["yolov8"] = YOLOv8Detector
        except:
            pass

        for name in detector_names:
            if name in detector_classes:
                try:
                    self.detectors[name] = detector_classes[name](
                        confidence_threshold=self.confidence_threshold *
                        0.8,  # Lower threshold for individual detectors
                        device=self.device
                    )
                    logger.info(f"Loaded {name} detector for ensemble")
                except Exception as e:
                    logger.warning(f"Failed to load {name} detector: {e}")

        if not self.detectors:
            raise RuntimeError("No detectors could be loaded for ensemble")

    def load_model(self):
        """Models are loaded individually by each detector."""
        pass

    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using ensemble voting."""
        all_detections = {}

        # Get detections from each detector
        for name, detector in self.detectors.items():
            try:
                detections = detector.detect(image)
                all_detections[name] = detections
                logger.debug(f"{name} found {len(detections)} faces")
            except Exception as e:
                logger.warning(f"Error in {name} detector: {e}")
                all_detections[name] = []

        # Apply voting mechanism
        final_faces = self._apply_voting(all_detections, image.shape[:2])
        return self.postprocess_results(final_faces)

    def _apply_voting(self, all_detections: Dict[str, List[FaceDetectionResult]],
                      image_shape: tuple) -> List[FaceDetectionResult]:
        """Apply voting mechanism to combine detections."""
        if not all_detections:
            return []

        # Flatten all detections
        all_faces = []
        for detector_name, faces in all_detections.items():
            for face in faces:
                all_faces.append((detector_name, face))

        if not all_faces:
            return []

        # Group faces by spatial proximity
        face_groups = self._group_nearby_faces(all_faces)

        # Apply voting within each group
        final_faces = []
        for group in face_groups:
            if len(group) >= self.voting_threshold:
                # Create consensus face from group
                consensus_face = self._create_consensus_face(group)
                if consensus_face:
                    final_faces.append(consensus_face)

        return final_faces

    def _group_nearby_faces(self, all_faces: List[tuple], iou_threshold: float = 0.3) -> List[List[tuple]]:
        """Group faces that are spatially close."""
        groups = []
        used = set()

        for i, (det_name, face) in enumerate(all_faces):
            if i in used:
                continue

            # Start new group
            group = [(det_name, face)]
            used.add(i)

            # Find nearby faces
            for j, (other_det_name, other_face) in enumerate(all_faces):
                if j in used or j == i:
                    continue

                # Calculate IoU
                iou = self._calculate_iou(face, other_face)
                if iou > iou_threshold:
                    group.append((other_det_name, other_face))
                    used.add(j)

            groups.append(group)

        return groups

    def _calculate_iou(self, face1: FaceDetectionResult, face2: FaceDetectionResult) -> float:
        """Calculate Intersection over Union between two faces."""
        # Get coordinates
        x1_1, y1_1, w1, h1 = face1.bbox
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = face2.bbox
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _create_consensus_face(self, group: List[tuple]) -> FaceDetectionResult:
        """Create consensus face from group of detections."""
        if not group:
            return None

        # Calculate average bounding box and confidence
        x_coords = []
        y_coords = []
        widths = []
        heights = []
        confidences = []
        methods = []

        for detector_name, face in group:
            x_coords.append(face.x)
            y_coords.append(face.y)
            widths.append(face.width)
            heights.append(face.height)
            confidences.append(face.confidence)
            methods.append(face.method)

        # Average coordinates
        avg_x = int(np.mean(x_coords))
        avg_y = int(np.mean(y_coords))
        avg_width = int(np.mean(widths))
        avg_height = int(np.mean(heights))

        # Weighted average confidence
        avg_confidence = np.mean(confidences)

        # Create method string
        method_str = f"ensemble({'+'.join(set(methods))})"

        return FaceDetectionResult(
            bbox=(avg_x, avg_y, avg_width, avg_height),
            confidence=avg_confidence,
            method=method_str
        )
