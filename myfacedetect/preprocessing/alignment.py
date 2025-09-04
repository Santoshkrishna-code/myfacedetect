"""
Face Alignment Module
Aligns faces based on facial landmarks for improved recognition accuracy.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class FaceAligner:
    """Face alignment using facial landmarks."""

    def __init__(self, desired_face_width: int = 224, desired_face_height: int = 224):
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        self.face_detector = None
        self.landmark_predictor = None
        self._load_models()

    def _load_models(self):
        """Load face detection and landmark models."""
        try:
            # Try to use dlib for landmark detection
            import dlib

            # Load dlib face detector
            self.face_detector = dlib.get_frontal_face_detector()

            # Try to load shape predictor
            try:
                # You would need to download this file
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                self.landmark_predictor = dlib.shape_predictor(predictor_path)
                logger.info("Dlib face alignment loaded successfully")
            except:
                logger.warning(
                    "Dlib shape predictor not found. Using fallback alignment.")
                self.landmark_predictor = None

        except ImportError:
            logger.info("Dlib not available, using MediaPipe for alignment")
            self._load_mediapipe_landmarks()

    def _load_mediapipe_landmarks(self):
        """Load MediaPipe face mesh for landmarks."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe face mesh loaded for alignment")
        except Exception as e:
            logger.warning(f"Could not load MediaPipe face mesh: {e}")
            self.mp_face_mesh = None

    def align_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Align face based on eye positions.

        Args:
            image: Input image
            bbox: Face bounding box (x, y, width, height)

        Returns:
            Aligned face image or None if alignment failed
        """
        try:
            # Extract face region
            x, y, w, h = bbox
            face_img = image[y:y+h, x:x+w]

            if self.landmark_predictor is not None:
                return self._align_with_dlib(image, bbox)
            elif self.mp_face_mesh is not None:
                return self._align_with_mediapipe(face_img)
            else:
                return self._simple_alignment(face_img)

        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return self._simple_alignment(face_img)

    def _align_with_dlib(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Align face using dlib landmarks."""
        import dlib

        x, y, w, h = bbox
        rect = dlib.rectangle(x, y, x + w, y + h)

        # Get landmarks
        landmarks = self.landmark_predictor(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), rect)

        # Extract eye coordinates (landmarks 36-41 for left eye, 42-47 for right eye)
        left_eye_pts = []
        right_eye_pts = []

        for i in range(36, 42):  # Left eye
            left_eye_pts.append([landmarks.part(i).x, landmarks.part(i).y])
        for i in range(42, 48):  # Right eye
            right_eye_pts.append([landmarks.part(i).x, landmarks.part(i).y])

        # Calculate eye centers
        left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)

        return self._align_by_eyes(image, left_eye_center, right_eye_center, bbox)

    def _align_with_mediapipe(self, face_img: np.ndarray) -> np.ndarray:
        """Align face using MediaPipe landmarks."""
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = face_img.shape[:2]

            # Get eye landmarks (approximate indices)
            left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133]
            right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263]

            # Calculate eye centers
            left_eye_x = np.mean(
                [landmarks.landmark[i].x * w for i in left_eye_idx])
            left_eye_y = np.mean(
                [landmarks.landmark[i].y * h for i in left_eye_idx])

            right_eye_x = np.mean(
                [landmarks.landmark[i].x * w for i in right_eye_idx])
            right_eye_y = np.mean(
                [landmarks.landmark[i].y * h for i in right_eye_idx])

            left_eye_center = [int(left_eye_x), int(left_eye_y)]
            right_eye_center = [int(right_eye_x), int(right_eye_y)]

            return self._align_by_eyes(face_img, left_eye_center, right_eye_center,
                                       (0, 0, face_img.shape[1], face_img.shape[0]))

        return self._simple_alignment(face_img)

    def _align_by_eyes(self, image: np.ndarray, left_eye: List[int], right_eye: List[int],
                       bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Align face based on eye positions."""
        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate center between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        # Apply rotation
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Extract and resize face
        x, y, w, h = bbox
        face_region = rotated[y:y+h, x:x+w]

        return cv2.resize(face_region, (self.desired_face_width, self.desired_face_height))

    def _simple_alignment(self, face_img: np.ndarray) -> np.ndarray:
        """Simple alignment - just resize."""
        return cv2.resize(face_img, (self.desired_face_width, self.desired_face_height))


def align_faces(images: List[np.ndarray], bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Align multiple faces.

    Args:
        images: List of input images
        bboxes: List of face bounding boxes

    Returns:
        List of aligned face images
    """
    aligner = FaceAligner()
    aligned_faces = []

    for image, bbox in zip(images, bboxes):
        aligned = aligner.align_face(image, bbox)
        if aligned is not None:
            aligned_faces.append(aligned)

    return aligned_faces
