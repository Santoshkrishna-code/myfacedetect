import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging

# Initialize mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetectionResult:
    """Class to hold face detection results with additional information."""

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float = 1.0, method: str = "unknown"):
        self.x, self.y, self.width, self.height = bbox
        self.confidence = confidence
        self.method = method

    def __repr__(self):
        return f"Face(x={self.x}, y={self.y}, w={self.width}, h={self.height}, conf={self.confidence:.2f}, method={self.method})"    @ property

    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of the face."""
        return (self.x + self.width // 2, self.y + self.height // 2)


def detect_faces(image_path: Union[str, Path],
                 method: str = "haar",
                 return_image: bool = False,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 4,
                 min_size: Tuple[int, int] = (30, 30)) -> Union[List[FaceDetectionResult], Tuple[List[FaceDetectionResult], np.ndarray]]:
    """
    Detect faces in an image with multiple detection methods.

    Args:
        image_path: Path to the image file or numpy array
        method: Detection method ('haar', 'mediapipe', or 'both')
        return_image: If True, return the image with face rectangles drawn
        scale_factor: Scale factor for Haar cascade (1.1 = 10% increase each scale)
        min_neighbors: Minimum neighbors for Haar cascade detection
        min_size: Minimum face size (width, height) in pixels

    Returns:
        List of FaceDetectionResult objects, optionally with annotated image

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If invalid detection method specified
    """
    if method not in ["haar", "mediapipe", "both"]:
        raise ValueError("Method must be 'haar', 'mediapipe', or 'both'")

    # Handle different input types
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        logger.info(f"Loaded image from {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path.copy()
        logger.info("Using provided numpy array as image")
    else:
        raise ValueError("image_path must be a file path or numpy array")

    faces = []

    if method in ["haar", "both"]:
        faces.extend(_detect_faces_haar(
            img, scale_factor, min_neighbors, min_size))

    if method in ["mediapipe", "both"]:
        faces.extend(_detect_faces_mediapipe(img))

    # Remove duplicates if using 'both' method
    if method == "both":
        faces = _remove_duplicate_faces(faces)

    logger.info(f"Detected {len(faces)} faces using {method} method")

    if return_image:
        annotated_img = _draw_faces(img.copy(), faces)
        return faces, annotated_img

    return faces


def _detect_faces_haar(img: np.ndarray,
                       scale_factor: float,
                       min_neighbors: int,
                       min_size: Tuple[int, int]) -> List[FaceDetectionResult]:
    """Detect faces using Haar cascades."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(
        gray, scale_factor, min_neighbors, minSize=min_size)

    return [FaceDetectionResult((int(x), int(y), int(w), int(h)), method="haar")
            for (x, y, w, h) in detected_faces]


def _detect_faces_mediapipe(img: np.ndarray) -> List[FaceDetectionResult]:
    """Detect faces using MediaPipe."""
    faces = []

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)

        if results.detections:
            h, w, _ = img.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]

                faces.append(FaceDetectionResult(
                    (x, y, width, height), confidence, "mediapipe"))

    return faces


def _remove_duplicate_faces(faces: List[FaceDetectionResult],
                            overlap_threshold: float = 0.3) -> List[FaceDetectionResult]:
    """Remove duplicate face detections based on overlap."""
    if len(faces) <= 1:
        return faces

    unique_faces = []
    for face in faces:
        is_duplicate = False
        for unique_face in unique_faces:
            if _calculate_overlap(face, unique_face) > overlap_threshold:
                is_duplicate = True
                # Keep the face with higher confidence
                if face.confidence > unique_face.confidence:
                    unique_faces.remove(unique_face)
                    unique_faces.append(face)
                break

        if not is_duplicate:
            unique_faces.append(face)

    return unique_faces


def _calculate_overlap(face1: FaceDetectionResult, face2: FaceDetectionResult) -> float:
    """Calculate overlap ratio between two face detections."""
    x1, y1, w1, h1 = face1.bbox
    x2, y2, w2, h2 = face2.bbox

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def _draw_faces(img: np.ndarray, faces: List[FaceDetectionResult]) -> np.ndarray:
    """Draw bounding boxes around detected faces."""
    for face in faces:
        x, y, w, h = face.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add confidence score if available
        if hasattr(face, 'confidence') and face.confidence < 1.0:
            cv2.putText(img, f'{face.confidence:.2f}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    return img


def detect_faces_realtime(camera_index: int = 0,
                          method: str = "mediapipe",
                          window_name: str = "Real-Time Face Detection",
                          show_fps: bool = True,
                          save_detections: bool = False,
                          output_dir: str = "detections") -> None:
    """
    Open webcam and detect faces in real-time with enhanced features.

    Args:
        camera_index: Index of the webcam (default=0)
        method: Detection method ('haar', 'mediapipe', or 'both')
        window_name: Name of the display window
        show_fps: Whether to display FPS counter
        save_detections: Whether to save screenshots with detected faces
        output_dir: Directory to save detections (if save_detections=True)

    Controls:
        ESC: Exit
        'c' or SPACE: Capture screenshot
        's': Toggle screenshot saving
        'f': Toggle FPS display
        'h': Switch to Haar cascade
        'm': Switch to MediaPipe
        'b': Switch to both methods
    """
    if method not in ["haar", "mediapipe", "both"]:
        raise ValueError("Method must be 'haar', 'mediapipe', or 'both'")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    # Create output directory if saving detections
    if save_detections:
        Path(output_dir).mkdir(exist_ok=True)

    # FPS calculation
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0

    screenshot_counter = 0
    current_method = method

    logger.info(f"Starting real-time detection with {current_method} method")
    logger.info("Press ESC to exit, 'c' to capture, 's' to toggle saving")

    # Initialize MediaPipe once for efficiency
    mp_face_detector = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) if current_method in ["mediapipe", "both"] else None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect faces based on current method
            if current_method == "haar":
                faces = _detect_faces_haar(frame, 1.1, 4, (30, 30))
            elif current_method == "mediapipe":
                if mp_face_detector:
                    with mp_face_detector:
                        faces = _detect_faces_mediapipe_realtime(
                            frame, mp_face_detector)
                else:
                    faces = []
            else:  # both
                haar_faces = _detect_faces_haar(frame, 1.1, 4, (30, 30))
                if mp_face_detector:
                    with mp_face_detector:
                        mp_faces = _detect_faces_mediapipe_realtime(
                            frame, mp_face_detector)
                else:
                    mp_faces = []
                faces = _remove_duplicate_faces(haar_faces + mp_faces)

            # Draw faces
            frame = _draw_faces_realtime(frame, faces, current_method)

            # Calculate and display FPS
            fps_counter += 1
            if fps_counter >= 30:  # Update every 30 frames
                fps_end_time = cv2.getTickCount()
                time_diff = (fps_end_time - fps_start_time) / \
                    cv2.getTickFrequency()
                current_fps = fps_counter / time_diff
                fps_counter = 0
                fps_start_time = cv2.getTickCount()

            # Display information
            if show_fps:
                cv2.putText(frame, f'FPS: {current_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f'Method: {current_method.upper()}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, f'Faces: {len(faces)}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display controls
            controls_text = "ESC:Exit | C:Capture | S:Save | F:FPS | H:Haar | M:MediaPipe | B:Both"
            cv2.putText(frame, controls_text, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow(window_name, frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                logger.info("Exiting real-time detection")
                break
            elif key == ord('c') or key == ord(' '):  # Capture
                screenshot_counter += 1
                filename = f"capture_{screenshot_counter:04d}.jpg"
                if save_detections:
                    filepath = Path(output_dir) / filename
                else:
                    filepath = filename
                cv2.imwrite(str(filepath), frame)
                logger.info(f"Screenshot saved: {filepath}")
            elif key == ord('s'):  # Toggle saving
                save_detections = not save_detections
                status = "ON" if save_detections else "OFF"
                logger.info(f"Screenshot saving: {status}")
                if save_detections:
                    Path(output_dir).mkdir(exist_ok=True)
            elif key == ord('f'):  # Toggle FPS
                show_fps = not show_fps
            elif key == ord('h'):  # Switch to Haar
                current_method = "haar"
                logger.info("Switched to Haar cascade method")
            elif key == ord('m'):  # Switch to MediaPipe
                current_method = "mediapipe"
                if not mp_face_detector:
                    mp_face_detector = mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5)
                logger.info("Switched to MediaPipe method")
            elif key == ord('b'):  # Switch to both
                current_method = "both"
                if not mp_face_detector:
                    mp_face_detector = mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5)
                logger.info("Switched to both methods")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")


def _detect_faces_mediapipe_realtime(frame: np.ndarray,
                                     face_detector) -> List[FaceDetectionResult]:
    """Optimized MediaPipe detection for real-time processing."""
    faces = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            confidence = detection.score[0]

            faces.append(FaceDetectionResult(
                (x, y, width, height), confidence, "mediapipe"))

    return faces


def _draw_faces_realtime(frame: np.ndarray,
                         faces: List[FaceDetectionResult],
                         method: str) -> np.ndarray:
    """Draw faces with method-specific colors and styles."""
    colors = {
        "haar": (255, 0, 0),      # Blue
        "mediapipe": (0, 255, 0),  # Green
        "both": (0, 255, 255)     # Yellow
    }

    color = colors.get(method, (255, 255, 255))

    for i, face in enumerate(faces):
        x, y, w, h = face.bbox

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw face number
        cv2.putText(frame, f'#{i+1}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw confidence if available
        if hasattr(face, 'confidence') and face.confidence < 1.0:
            cv2.putText(frame, f'{face.confidence:.2f}',
                        (x + w - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        # Draw center point
        center = face.center
        cv2.circle(frame, center, 3, color, -1)

    return frame


# Utility functions for advanced features
def batch_detect_faces(image_paths: List[Union[str, Path]],
                       method: str = "haar",
                       **kwargs) -> List[List[FaceDetectionResult]]:
    """
    Detect faces in multiple images efficiently.

    Args:
        image_paths: List of image paths
        method: Detection method
        **kwargs: Additional arguments for detect_faces

    Returns:
        List of detection results for each image
    """
    results = []
    for path in image_paths:
        try:
            faces = detect_faces(path, method=method, **kwargs)
            results.append(faces)
            logger.info(f"Processed {path}: {len(faces)} faces")
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            results.append([])

    return results


def save_face_crops(image_path: Union[str, Path],
                    output_dir: str = "face_crops",
                    method: str = "haar") -> List[str]:
    """
    Detect faces and save individual face crops.

    Args:
        image_path: Path to input image
        output_dir: Directory to save face crops
        method: Detection method

    Returns:
        List of saved crop filenames
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    faces = detect_faces(image_path, method=method)

    Path(output_dir).mkdir(exist_ok=True)
    saved_files = []

    for i, face in enumerate(faces):
        x, y, w, h = face.bbox
        face_crop = img[y:y+h, x:x+w]

        base_name = Path(image_path).stem
        crop_filename = f"{base_name}_face_{i+1}.jpg"
        crop_path = Path(output_dir) / crop_filename

        cv2.imwrite(str(crop_path), face_crop)
        saved_files.append(str(crop_path))
        logger.info(f"Saved face crop: {crop_path}")

    return saved_files
