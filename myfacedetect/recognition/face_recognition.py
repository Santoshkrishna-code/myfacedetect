"""
Face Recognition Module
Advanced face recognition using deep learning embeddings.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import pickle
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognition using deep learning embeddings."""

    def __init__(self, model_name: str = 'arcface', embedding_dim: int = 512):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.known_faces = {}  # {name: embeddings_list}
        self.face_database_path = "face_database.pkl"
        self.metadata_path = "face_metadata.json"
        self.threshold = 0.6  # Similarity threshold

        self._load_model()
        self._load_database()

    def _load_model(self):
        """Load face recognition model."""
        try:
            if self.model_name == 'arcface':
                self._load_arcface_model()
            elif self.model_name == 'facenet':
                self._load_facenet_model()
            elif self.model_name == 'opencv':
                self._load_opencv_model()
            else:
                logger.warning(
                    f"Unknown model: {self.model_name}, using OpenCV")
                self._load_opencv_model()
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self._load_opencv_model()

    def _load_arcface_model(self):
        """Load ArcFace model using InsightFace with CPU-only configuration."""
        try:
            import insightface
            import os

            # Force CPU mode to avoid CUDA errors
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            self.model = insightface.app.FaceAnalysis(
                providers=['CPUExecutionProvider']
            )
            # Force CPU with ctx_id=-1
            self.model.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("ArcFace model loaded successfully on CPU")

        except ImportError:
            logger.warning("InsightFace not available, falling back to OpenCV")
            self._load_opencv_model()
        except Exception as e:
            logger.warning(f"ArcFace model loading failed: {e}")
            self._load_opencv_model()

    def _load_facenet_model(self):
        """Load FaceNet model."""
        try:
            # This would require tensorflow and a FaceNet model
            # For now, fall back to OpenCV
            logger.info("FaceNet model not implemented, using OpenCV")
            self._load_opencv_model()
        except Exception as e:
            logger.warning(f"FaceNet model loading failed: {e}")
            self._load_opencv_model()

    def _load_opencv_model(self):
        """Load OpenCV face recognizer as fallback."""
        try:
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.model_name = 'opencv'
            logger.info("OpenCV LBPH face recognizer loaded")
        except AttributeError:
            logger.error("OpenCV face module not available")
            self.model = None

    def _load_database(self):
        """Load face database from file."""
        try:
            if os.path.exists(self.face_database_path):
                with open(self.face_database_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces")

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}

        except Exception as e:
            logger.warning(f"Could not load face database: {e}")
            self.known_faces = {}
            self.metadata = {}

    def _save_database(self):
        """Save face database to file."""
        try:
            with open(self.face_database_path, 'wb') as f:
                pickle.dump(self.known_faces, f)

            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            logger.info("Face database saved successfully")
        except Exception as e:
            logger.error(f"Could not save face database: {e}")

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from image.

        Args:
            face_image: Aligned face image

        Returns:
            Face embedding vector or None if failed
        """
        try:
            if self.model is None:
                logger.warning("Recognition model not loaded")
                return None

            # Validate input image
            if face_image is None or face_image.size == 0:
                logger.warning("Invalid face image provided")
                return None

            # Check image dimensions - should be at least 32x32
            if face_image.shape[0] < 32 or face_image.shape[1] < 32:
                logger.warning(f"Face image too small: {face_image.shape}")
                return None

            if self.model_name == 'arcface':
                embedding = self._get_arcface_embedding(face_image)
                if embedding is not None and len(embedding) > 0:
                    return embedding
                else:
                    logger.debug(
                        "ArcFace embedding extraction returned empty result")
                    return None
            elif self.model_name == 'opencv':
                return self._get_opencv_embedding(face_image)
            else:
                logger.warning(f"Unknown model name: {self.model_name}")
                return None

        except Exception as e:
            logger.warning(
                f"Embedding extraction failed for {self.model_name}: {e}")
            return None

    def _get_arcface_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding using ArcFace model."""
        try:
            if self.model is None:
                logger.warning("ArcFace model not loaded")
                return None

            # Validate image
            if face_image is None or face_image.size == 0:
                logger.debug("Invalid face image for ArcFace")
                return None

            # ArcFace expects BGR image with specific format
            if len(face_image.shape) == 3:
                bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

            # Ensure minimum size for ArcFace
            if bgr_image.shape[0] < 64 or bgr_image.shape[1] < 64:
                logger.debug(
                    f"Resizing small image {bgr_image.shape} for ArcFace")
                bgr_image = cv2.resize(bgr_image, (112, 112))

            faces = self.model.get(bgr_image)
            if faces and len(faces) > 0:
                embedding = faces[0].embedding
                if embedding is not None and len(embedding) > 0:
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    return embedding
                else:
                    logger.debug("ArcFace returned empty embedding")
                    return None
            else:
                logger.debug(
                    "ArcFace: No faces detected for embedding extraction")
                return None

        except Exception as e:
            logger.debug(f"ArcFace embedding failed: {e}")
            return None

    def _get_opencv_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding using OpenCV (simplified)."""
        try:
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image

            # Resize to standard size
            gray = cv2.resize(gray, (100, 100))

            # Flatten as simple embedding
            return gray.flatten().astype(np.float32) / 255.0
        except Exception as e:
            logger.warning(f"OpenCV embedding failed: {e}")
            return None

    def add_face(self, face_image: np.ndarray, name: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a new face to the database.

        Args:
            face_image: Aligned face image
            name: Person's name
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            embedding = self.get_embedding(face_image)
            if embedding is None:
                return False

            # Initialize person if not exists
            if name not in self.known_faces:
                self.known_faces[name] = []

            # Add embedding
            self.known_faces[name].append(embedding)

            # Add metadata
            if name not in self.metadata:
                self.metadata[name] = []

            face_metadata = {
                'added_date': datetime.now().isoformat(),
                'embedding_dim': len(embedding),
                'model': self.model_name
            }
            if metadata:
                face_metadata.update(metadata)

            self.metadata[name].append(face_metadata)

            # Save database
            self._save_database()

            logger.info(f"Added face for {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add face for {name}: {e}")
            return False

    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face from the database.

        Args:
            face_image: Aligned face image

        Returns:
            (name, confidence) tuple. name is None if no match found.
        """
        try:
            embedding = self.get_embedding(face_image)
            if embedding is None:
                return None, 0.0

            best_match = None
            best_similarity = 0.0

            # Compare with all known faces
            for name, embeddings in self.known_faces.items():
                for known_embedding in embeddings:
                    similarity = self._calculate_similarity(
                        embedding, known_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name

            # Check if similarity is above threshold
            if best_similarity >= self.threshold:
                return best_match, best_similarity
            else:
                return None, best_similarity

        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two embeddings."""
        try:
            if self.model_name == 'arcface':
                # Cosine similarity for ArcFace
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
                return (cosine_sim + 1) / 2  # Normalize to [0, 1]
            else:
                # Euclidean distance for others (convert to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                return 1.0 / (1.0 + distance)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Public method to compare two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0.0 to 1.0)
        """
        return self._calculate_similarity(embedding1, embedding2)

    def get_all_names(self) -> List[str]:
        """Get all known person names."""
        return list(self.known_faces.keys())

    def remove_person(self, name: str) -> bool:
        """
        Remove a person from the database.

        Args:
            name: Person's name to remove

        Returns:
            True if successful, False otherwise
        """
        try:
            if name in self.known_faces:
                del self.known_faces[name]
                if name in self.metadata:
                    del self.metadata[name]
                self._save_database()
                logger.info(f"Removed {name} from database")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove {name}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_people': len(self.known_faces),
            'total_faces': sum(len(embeddings) for embeddings in self.known_faces.values()),
            'model': self.model_name,
            'embedding_dim': self.embedding_dim,
            'threshold': self.threshold
        }

        if self.known_faces:
            faces_per_person = [len(embeddings)
                                for embeddings in self.known_faces.values()]
            stats['avg_faces_per_person'] = np.mean(faces_per_person)
            stats['min_faces_per_person'] = min(faces_per_person)
            stats['max_faces_per_person'] = max(faces_per_person)

        return stats

    def set_threshold(self, threshold: float):
        """Set recognition threshold."""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Recognition threshold set to {self.threshold}")


def create_face_recognizer(model_name: str = 'arcface') -> FaceRecognizer:
    """
    Create a face recognizer instance.

    Args:
        model_name: Model to use ('arcface', 'facenet', 'opencv')

    Returns:
        FaceRecognizer instance
    """
    return FaceRecognizer(model_name=model_name)
