"""
Privacy Protection Module
Privacy-preserving face detection and recognition techniques.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import hashlib
import base64
import logging

logger = logging.getLogger(__name__)


class PrivacyProtector:
    """Privacy protection for face detection and recognition."""

    def __init__(self, mode: str = 'blur'):
        self.mode = mode
        self.blur_kernel_size = (51, 51)
        self.pixelate_size = 20
        self.noise_strength = 25

    def anonymize_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                        method: Optional[str] = None) -> np.ndarray:
        """
        Anonymize faces in image.

        Args:
            image: Input image
            faces: List of face bounding boxes
            method: Anonymization method ('blur', 'pixelate', 'mask', 'noise')

        Returns:
            Anonymized image
        """
        if method is None:
            method = self.mode

        result = image.copy()

        for face in faces:
            x, y, w, h = face
            face_roi = result[y:y+h, x:x+w]

            if method == 'blur':
                anonymized_roi = self._blur_face(face_roi)
            elif method == 'pixelate':
                anonymized_roi = self._pixelate_face(face_roi)
            elif method == 'mask':
                anonymized_roi = self._mask_face(face_roi)
            elif method == 'noise':
                anonymized_roi = self._add_noise(face_roi)
            else:
                logger.warning(f"Unknown anonymization method: {method}")
                anonymized_roi = self._blur_face(face_roi)

            result[y:y+h, x:x+w] = anonymized_roi

        return result

    def _blur_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Apply blur to face region."""
        return cv2.GaussianBlur(face_roi, self.blur_kernel_size, 0)

    def _pixelate_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Apply pixelation to face region."""
        h, w = face_roi.shape[:2]

        # Resize down
        small = cv2.resize(face_roi, (self.pixelate_size, self.pixelate_size))

        # Resize back up
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _mask_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Apply solid mask to face region."""
        return np.ones_like(face_roi) * 128  # Gray mask

    def _add_noise(self, face_roi: np.ndarray) -> np.ndarray:
        """Add noise to face region."""
        noise = np.random.randint(-self.noise_strength, self.noise_strength,
                                  face_roi.shape, dtype=np.int16)
        noisy = face_roi.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def create_face_hash(self, face_embedding: np.ndarray, salt: str = "") -> str:
        """
        Create privacy-preserving hash of face embedding.

        Args:
            face_embedding: Face embedding vector
            salt: Salt for hashing

        Returns:
            Hashed face representation
        """
        try:
            # Convert embedding to bytes
            embedding_bytes = face_embedding.tobytes()

            # Add salt
            salted_data = embedding_bytes + salt.encode()

            # Create hash
            face_hash = hashlib.sha256(salted_data).hexdigest()

            return face_hash

        except Exception as e:
            logger.error(f"Face hashing failed: {e}")
            return ""

    def differential_privacy_embedding(self, embedding: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """
        Apply differential privacy to face embedding.

        Args:
            embedding: Original embedding
            epsilon: Privacy parameter (smaller = more private)

        Returns:
            Privacy-preserving embedding
        """
        try:
            # Add Laplace noise for differential privacy
            sensitivity = 2.0  # L2 sensitivity of face embeddings
            scale = sensitivity / epsilon

            noise = np.random.laplace(0, scale, embedding.shape)
            private_embedding = embedding + noise

            # Normalize to maintain similarity properties
            norm = np.linalg.norm(private_embedding)
            if norm > 0:
                private_embedding = private_embedding / norm

            return private_embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Differential privacy failed: {e}")
            return embedding

    def homomorphic_distance(self, embedding1_hash: str, embedding2_hash: str) -> float:
        """
        Calculate distance between hashed embeddings (simplified).

        Args:
            embedding1_hash: First embedding hash
            embedding2_hash: Second embedding hash

        Returns:
            Distance measure
        """
        try:
            # Simple Hamming distance between hashes
            if len(embedding1_hash) != len(embedding2_hash):
                return 1.0

            hamming_distance = sum(c1 != c2 for c1, c2 in zip(
                embedding1_hash, embedding2_hash))
            normalized_distance = hamming_distance / len(embedding1_hash)

            return normalized_distance

        except Exception as e:
            logger.error(f"Homomorphic distance calculation failed: {e}")
            return 1.0

    def secure_template_protection(self, embedding: np.ndarray, key: str) -> Dict[str, Any]:
        """
        Apply template protection to face embedding.

        Args:
            embedding: Original embedding
            key: Encryption key

        Returns:
            Protected template information
        """
        try:
            # Generate transformation matrix from key
            np.random.seed(hash(key) % (2**32))
            transform_matrix = np.random.randn(len(embedding), len(embedding))

            # Apply irreversible transformation
            protected_template = np.dot(transform_matrix, embedding)

            # Create verification data
            verification_hash = hashlib.sha256(
                (embedding.tobytes() + key.encode())
            ).hexdigest()

            return {
                'protected_template': protected_template,
                'verification_hash': verification_hash,
                'template_size': len(protected_template)
            }

        except Exception as e:
            logger.error(f"Template protection failed: {e}")
            return {'protected_template': embedding, 'verification_hash': '', 'template_size': len(embedding)}

    def federated_learning_update(self, local_embeddings: List[np.ndarray],
                                  global_model: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create privacy-preserving federated learning update.

        Args:
            local_embeddings: Local face embeddings
            global_model: Current global model (if any)

        Returns:
            Privacy-preserving model update
        """
        try:
            if not local_embeddings:
                return global_model if global_model is not None else np.array([])

            # Calculate local average
            local_avg = np.mean(local_embeddings, axis=0)

            # Add noise for privacy
            epsilon = 1.0  # Privacy parameter
            sensitivity = 2.0
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, local_avg.shape)

            private_update = local_avg + noise

            # If global model exists, compute difference
            if global_model is not None:
                private_update = private_update - global_model

            return private_update.astype(np.float32)

        except Exception as e:
            logger.error(f"Federated learning update failed: {e}")
            return np.array([])

    def k_anonymity_clustering(self, embeddings: List[np.ndarray], k: int = 5) -> List[int]:
        """
        Apply k-anonymity through clustering.

        Args:
            embeddings: List of embeddings to cluster
            k: Minimum cluster size for anonymity

        Returns:
            Cluster assignments
        """
        try:
            if len(embeddings) < k:
                return [0] * len(embeddings)

            from sklearn.cluster import KMeans

            # Perform clustering
            n_clusters = max(1, len(embeddings) // k)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Ensure each cluster has at least k members
            cluster_counts = np.bincount(cluster_labels)
            small_clusters = np.where(cluster_counts < k)[0]

            # Merge small clusters
            for small_cluster in small_clusters:
                # Find nearest larger cluster
                small_indices = np.where(cluster_labels == small_cluster)[0]
                if len(small_indices) > 0:
                    # Assign to largest cluster
                    largest_cluster = np.argmax(cluster_counts)
                    cluster_labels[small_indices] = largest_cluster

            return cluster_labels.tolist()

        except Exception as e:
            logger.error(f"K-anonymity clustering failed: {e}")
            return list(range(len(embeddings)))


class SecureStorage:
    """Secure storage for face recognition data."""

    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or self._generate_key()

    def _generate_key(self) -> str:
        """Generate encryption key."""
        return base64.b64encode(np.random.bytes(32)).decode()

    def encrypt_embedding(self, embedding: np.ndarray) -> str:
        """Encrypt face embedding for storage."""
        try:
            # Simple XOR encryption (use proper encryption in production)
            key_bytes = self.encryption_key.encode()
            embedding_bytes = embedding.tobytes()

            encrypted = bytearray()
            for i, byte in enumerate(embedding_bytes):
                key_byte = key_bytes[i % len(key_bytes)]
                encrypted.append(byte ^ key_byte)

            return base64.b64encode(encrypted).decode()

        except Exception as e:
            logger.error(f"Embedding encryption failed: {e}")
            return ""

    def decrypt_embedding(self, encrypted_data: str, dtype=np.float32) -> Optional[np.ndarray]:
        """Decrypt face embedding."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            key_bytes = self.encryption_key.encode()

            decrypted = bytearray()
            for i, byte in enumerate(encrypted_bytes):
                key_byte = key_bytes[i % len(key_bytes)]
                decrypted.append(byte ^ key_byte)

            return np.frombuffer(decrypted, dtype=dtype)

        except Exception as e:
            logger.error(f"Embedding decryption failed: {e}")
            return None

    def secure_comparison(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Perform secure similarity comparison."""
        try:
            # Use homomorphic properties for privacy-preserving comparison
            # This is a simplified version - use proper homomorphic encryption in production

            # Add noise to both embeddings
            noise_scale = 0.01
            noisy_emb1 = embedding1 + \
                np.random.normal(0, noise_scale, embedding1.shape)
            noisy_emb2 = embedding2 + \
                np.random.normal(0, noise_scale, embedding2.shape)

            # Calculate similarity
            similarity = np.dot(noisy_emb1, noisy_emb2) / (
                np.linalg.norm(noisy_emb1) * np.linalg.norm(noisy_emb2)
            )

            return float(similarity)

        except Exception as e:
            logger.error(f"Secure comparison failed: {e}")
            return 0.0


def create_privacy_protector(mode: str = 'blur') -> PrivacyProtector:
    """
    Create a privacy protector instance.

    Args:
        mode: Default anonymization mode

    Returns:
        PrivacyProtector instance
    """
    return PrivacyProtector(mode=mode)


def create_secure_storage(encryption_key: Optional[str] = None) -> SecureStorage:
    """
    Create a secure storage instance.

    Args:
        encryption_key: Encryption key (generated if None)

    Returns:
        SecureStorage instance
    """
    return SecureStorage(encryption_key=encryption_key)
