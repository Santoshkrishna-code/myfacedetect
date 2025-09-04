"""
Liveness Detection Module
Anti-spoofing and liveness detection for face recognition security.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class LivenessDetector:
    """Liveness detection to prevent spoofing attacks."""

    def __init__(self, method: str = 'blink_eye_motion'):
        self.method = method
        self.challenge_timeout = 10.0  # seconds
        self.blink_threshold = 0.2
        self.motion_threshold = 5.0
        self.frame_buffer_size = 30

        # Initialize detection components
        self._load_models()
        self._reset_state()

    def _load_models(self):
        """Load required models for liveness detection."""
        try:
            if 'eye' in self.method or 'blink' in self.method:
                # Load eye cascade for blink detection
                eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

            if 'face' in self.method:
                # Load face cascade
                face_cascade_path = cv2.data.haarcascades + \
                    'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

            # Try to load advanced models
            self._load_advanced_models()

        except Exception as e:
            logger.warning(f"Could not load all liveness models: {e}")

    def _load_advanced_models(self):
        """Load advanced liveness detection models."""
        try:
            # Try to load MediaPipe for face mesh
            import mediapipe as mp

            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )

            logger.info(
                "MediaPipe models loaded for advanced liveness detection")

        except ImportError:
            logger.info(
                "MediaPipe not available for advanced liveness detection")
            self.mp_face_mesh = None
            self.mp_face_detection = None

    def _reset_state(self):
        """Reset detection state."""
        self.frame_buffer = deque(maxlen=self.frame_buffer_size)
        self.eye_aspect_ratios = deque(maxlen=20)
        self.motion_history = deque(maxlen=10)
        self.blink_count = 0
        self.start_time = time.time()
        self.challenge_active = False
        self.current_challenge = None

    def start_liveness_check(self, challenge_type: str = 'auto') -> Dict[str, Any]:
        """
        Start a liveness detection challenge.

        Args:
            challenge_type: Type of challenge ('blink', 'smile', 'turn_head', 'auto')

        Returns:
            Challenge information
        """
        self._reset_state()
        self.challenge_active = True

        if challenge_type == 'auto':
            # Choose challenge based on available capabilities
            if self.mp_face_mesh:
                challenge_type = np.random.choice(
                    ['blink', 'smile', 'turn_head'])
            else:
                challenge_type = 'blink'

        self.current_challenge = {
            'type': challenge_type,
            'start_time': time.time(),
            'timeout': self.challenge_timeout,
            'status': 'active',
            'progress': 0.0
        }

        return self.current_challenge

    def process_frame(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Process a frame for liveness detection.

        Args:
            frame: Input frame
            face_bbox: Face bounding box if known

        Returns:
            Detection results
        """
        if not self.challenge_active:
            return {'status': 'inactive', 'liveness_score': 0.0}

        # Check timeout
        if time.time() - self.current_challenge['start_time'] > self.challenge_timeout:
            self.challenge_active = False
            return {'status': 'timeout', 'liveness_score': 0.0}

        # Add frame to buffer
        self.frame_buffer.append(frame.copy())

        # Process based on challenge type
        if self.current_challenge['type'] == 'blink':
            return self._process_blink_challenge(frame, face_bbox)
        elif self.current_challenge['type'] == 'smile':
            return self._process_smile_challenge(frame, face_bbox)
        elif self.current_challenge['type'] == 'turn_head':
            return self._process_head_turn_challenge(frame, face_bbox)
        elif self.current_challenge['type'] == 'motion':
            return self._process_motion_challenge(frame, face_bbox)
        else:
            return self._process_general_liveness(frame, face_bbox)

    def _process_blink_challenge(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Process blink detection challenge."""
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                face_roi = frame[y:y+h, x:x+w]
            else:
                face_roi = frame

            # Detect eyes
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)

            if len(eyes) >= 2:
                # Calculate eye aspect ratio
                ear = self._calculate_eye_aspect_ratio(face_roi, eyes)
                self.eye_aspect_ratios.append(ear)

                # Detect blink
                if len(self.eye_aspect_ratios) > 5:
                    if ear < self.blink_threshold and np.mean(list(self.eye_aspect_ratios)[-3:]) < self.blink_threshold:
                        if len(self.eye_aspect_ratios) > 10 and np.mean(list(self.eye_aspect_ratios)[-10:-5]) > self.blink_threshold:
                            self.blink_count += 1

                # Check success condition
                required_blinks = 2
                if self.blink_count >= required_blinks:
                    self.challenge_active = False
                    return {
                        'status': 'success',
                        'liveness_score': 1.0,
                        'challenge': 'blink',
                        'blinks_detected': self.blink_count
                    }

                progress = min(self.blink_count / required_blinks, 1.0)
                return {
                    'status': 'in_progress',
                    'liveness_score': progress,
                    'challenge': 'blink',
                    'progress': progress,
                    'instruction': f"Please blink {required_blinks - self.blink_count} more time(s)"
                }

            return {
                'status': 'no_eyes_detected',
                'liveness_score': 0.0,
                'instruction': "Please look at the camera"
            }

        except Exception as e:
            logger.warning(f"Blink detection failed: {e}")
            return {'status': 'error', 'liveness_score': 0.0}

    def _process_smile_challenge(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Process smile detection challenge."""
        try:
            if not self.mp_face_mesh:
                return {'status': 'not_supported', 'liveness_score': 0.0}

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]

                # Get mouth landmarks for smile detection
                smile_score = self._calculate_smile_score(landmarks, w, h)

                if smile_score > 0.7:  # Threshold for smile detection
                    self.challenge_active = False
                    return {
                        'status': 'success',
                        'liveness_score': 1.0,
                        'challenge': 'smile',
                        'smile_score': smile_score
                    }

                return {
                    'status': 'in_progress',
                    'liveness_score': smile_score,
                    'challenge': 'smile',
                    'progress': smile_score,
                    'instruction': "Please smile"
                }

            return {
                'status': 'no_face_detected',
                'liveness_score': 0.0,
                'instruction': "Please look at the camera"
            }

        except Exception as e:
            logger.warning(f"Smile detection failed: {e}")
            return {'status': 'error', 'liveness_score': 0.0}

    def _process_head_turn_challenge(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Process head turn challenge."""
        try:
            if not self.mp_face_mesh:
                return {'status': 'not_supported', 'liveness_score': 0.0}

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]

                # Calculate head pose
                head_pose = self._calculate_head_pose(landmarks, w, h)

                # Check for significant head movement
                if abs(head_pose['yaw']) > 20 or abs(head_pose['pitch']) > 15:
                    self.challenge_active = False
                    return {
                        'status': 'success',
                        'liveness_score': 1.0,
                        'challenge': 'turn_head',
                        'head_pose': head_pose
                    }

                progress = (abs(head_pose['yaw']) +
                            abs(head_pose['pitch'])) / 35.0
                return {
                    'status': 'in_progress',
                    'liveness_score': min(progress, 0.9),
                    'challenge': 'turn_head',
                    'progress': progress,
                    'instruction': "Please turn your head slightly"
                }

            return {
                'status': 'no_face_detected',
                'liveness_score': 0.0,
                'instruction': "Please look at the camera"
            }

        except Exception as e:
            logger.warning(f"Head turn detection failed: {e}")
            return {'status': 'error', 'liveness_score': 0.0}

    def _process_motion_challenge(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Process motion-based liveness detection."""
        try:
            if len(self.frame_buffer) < 2:
                return {
                    'status': 'collecting_frames',
                    'liveness_score': 0.0,
                    'instruction': "Stay still for a moment"
                }

            # Calculate motion between frames
            prev_frame = self.frame_buffer[-2]
            curr_frame = self.frame_buffer[-1]

            motion_score = self._calculate_motion_score(
                prev_frame, curr_frame, face_bbox)
            self.motion_history.append(motion_score)

            # Check for natural micro-movements
            if len(self.motion_history) >= 5:
                avg_motion = np.mean(self.motion_history)
                motion_variance = np.var(self.motion_history)

                # Good liveness: some motion but not too much, with natural variation
                if 1.0 < avg_motion < 8.0 and motion_variance > 0.5:
                    self.challenge_active = False
                    return {
                        'status': 'success',
                        'liveness_score': 1.0,
                        'challenge': 'motion',
                        'motion_score': avg_motion
                    }
                elif avg_motion < 0.5:
                    return {
                        'status': 'too_still',
                        'liveness_score': 0.2,
                        'instruction': "Move slightly or blink"
                    }
                elif avg_motion > 15.0:
                    return {
                        'status': 'too_much_motion',
                        'liveness_score': 0.1,
                        'instruction': "Please stay more still"
                    }

            return {
                'status': 'in_progress',
                'liveness_score': 0.5,
                'challenge': 'motion',
                'instruction': "Natural movements detected..."
            }

        except Exception as e:
            logger.warning(f"Motion detection failed: {e}")
            return {'status': 'error', 'liveness_score': 0.0}

    def _process_general_liveness(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """General liveness detection combining multiple methods."""
        try:
            scores = []

            # Motion analysis
            if len(self.frame_buffer) >= 2:
                motion_result = self._process_motion_challenge(
                    frame, face_bbox)
                if motion_result['status'] == 'success':
                    scores.append(0.4)
                elif motion_result['liveness_score'] > 0:
                    scores.append(motion_result['liveness_score'] * 0.4)

            # Texture analysis (basic)
            texture_score = self._analyze_texture(frame, face_bbox)
            scores.append(texture_score * 0.3)

            # Color analysis
            color_score = self._analyze_color_distribution(frame, face_bbox)
            scores.append(color_score * 0.3)

            total_score = sum(scores)

            if total_score > 0.8:
                self.challenge_active = False
                return {
                    'status': 'success',
                    'liveness_score': total_score,
                    'challenge': 'general'
                }

            return {
                'status': 'in_progress',
                'liveness_score': total_score,
                'challenge': 'general',
                'progress': total_score
            }

        except Exception as e:
            logger.warning(f"General liveness detection failed: {e}")
            return {'status': 'error', 'liveness_score': 0.0}

    def _calculate_eye_aspect_ratio(self, face_roi: np.ndarray, eyes: np.ndarray) -> float:
        """Calculate eye aspect ratio for blink detection."""
        try:
            if len(eyes) < 2:
                return 1.0

            # Simple approximation using eye bounding boxes
            eye1 = eyes[0]
            eye2 = eyes[1]

            # Calculate aspect ratios
            ear1 = eye1[3] / eye1[2] if eye1[2] > 0 else 1.0
            ear2 = eye2[3] / eye2[2] if eye2[2] > 0 else 1.0

            return (ear1 + ear2) / 2.0

        except Exception as e:
            logger.warning(f"EAR calculation failed: {e}")
            return 1.0

    def _calculate_smile_score(self, landmarks, width: int, height: int) -> float:
        """Calculate smile score from facial landmarks."""
        try:
            # Mouth corner and center points (approximate indices)
            left_corner = [landmarks.landmark[61].x *
                           width, landmarks.landmark[61].y * height]
            right_corner = [landmarks.landmark[291].x *
                            width, landmarks.landmark[291].y * height]
            top_lip = [landmarks.landmark[13].x * width,
                       landmarks.landmark[13].y * height]
            bottom_lip = [landmarks.landmark[14].x *
                          width, landmarks.landmark[14].y * height]

            # Calculate mouth width and height
            mouth_width = abs(right_corner[0] - left_corner[0])
            mouth_height = abs(bottom_lip[1] - top_lip[1])

            # Simple smile detection based on width/height ratio
            if mouth_height > 0:
                ratio = mouth_width / mouth_height
                # Normalize to 0-1 score
                return min(max((ratio - 3.0) / 2.0, 0.0), 1.0)

            return 0.0

        except Exception as e:
            logger.warning(f"Smile score calculation failed: {e}")
            return 0.0

    def _calculate_head_pose(self, landmarks, width: int, height: int) -> Dict[str, float]:
        """Calculate head pose from facial landmarks."""
        try:
            # Key points for pose estimation
            nose_tip = [landmarks.landmark[1].x *
                        width, landmarks.landmark[1].y * height]
            left_eye = [landmarks.landmark[33].x *
                        width, landmarks.landmark[33].y * height]
            right_eye = [landmarks.landmark[263].x *
                         width, landmarks.landmark[263].y * height]

            # Calculate angles (simplified)
            eye_center = [(left_eye[0] + right_eye[0]) / 2,
                          (left_eye[1] + right_eye[1]) / 2]

            # Yaw: nose position relative to eye center
            yaw = (nose_tip[0] - eye_center[0]) / width * 180

            # Pitch: nose position relative to eye center
            pitch = (nose_tip[1] - eye_center[1]) / height * 180

            return {'yaw': yaw, 'pitch': pitch, 'roll': 0.0}

        except Exception as e:
            logger.warning(f"Head pose calculation failed: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

    def _calculate_motion_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                                face_bbox: Optional[Tuple[int, int, int, int]]) -> float:
        """Calculate motion score between frames."""
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Focus on face region if available
            if face_bbox:
                x, y, w, h = face_bbox
                prev_gray = prev_gray[y:y+h, x:x+w]
                curr_gray = curr_gray[y:y+h, x:x+w]

            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray,
                cv2.goodFeaturesToTrack(prev_gray, 100, 0.3, 7),
                None
            )[1]

            if flow is not None and len(flow) > 0:
                # Calculate motion magnitude
                motion = np.mean(np.linalg.norm(flow, axis=1))
                return float(motion)

            return 0.0

        except Exception as e:
            logger.warning(f"Motion score calculation failed: {e}")
            return 0.0

    def _analyze_texture(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> float:
        """Analyze texture to detect fake faces."""
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                face_roi = frame[y:y+h, x:x+w]
            else:
                face_roi = frame

            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Calculate local binary patterns
            lbp = self._calculate_lbp(gray)

            # Calculate texture variance
            variance = np.var(lbp)

            # Normalize score (real faces typically have higher texture variance)
            score = min(variance / 1000.0, 1.0)
            return score

        except Exception as e:
            logger.warning(f"Texture analysis failed: {e}")
            return 0.5

    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        try:
            height, width = image.shape
            lbp = np.zeros((height - 2, width - 2), dtype=np.uint8)

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    center = image[i, j]
                    pattern = 0

                    # Compare with 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]

                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            pattern |= (1 << k)

                    lbp[i-1, j-1] = pattern

            return lbp

        except Exception as e:
            logger.warning(f"LBP calculation failed: {e}")
            return image

    def _analyze_color_distribution(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> float:
        """Analyze color distribution for liveness detection."""
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                face_roi = frame[y:y+h, x:x+w]
            else:
                face_roi = frame

            # Convert to different color spaces
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)

            # Calculate color variance in different channels
            hsv_var = np.var(hsv, axis=(0, 1))
            lab_var = np.var(lab, axis=(0, 1))

            # Combine variances (real faces have natural color variation)
            total_variance = np.sum(hsv_var) + np.sum(lab_var)

            # Normalize score
            score = min(total_variance / 10000.0, 1.0)
            return score

        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return 0.5


def create_liveness_detector(method: str = 'blink_eye_motion') -> LivenessDetector:
    """
    Create a liveness detector instance.

    Args:
        method: Detection method to use

    Returns:
        LivenessDetector instance
    """
    return LivenessDetector(method=method)
