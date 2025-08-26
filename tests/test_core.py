"""
Unit tests for core face detection functionality.
"""
import pytest
import numpy as np
from pathlib import Path

from myfacedetect.core import (
    detect_faces, FaceDetectionResult,
    _detect_faces_haar, _detect_faces_mediapipe,
    _remove_duplicate_faces, _calculate_overlap
)


class TestFaceDetectionResult:
    """Test FaceDetectionResult class."""

    def test_initialization(self):
        """Test basic initialization."""
        face = FaceDetectionResult((10, 20, 50, 60), 0.95)
        assert face.x == 10
        assert face.y == 20
        assert face.width == 50
        assert face.height == 60
        assert face.confidence == 0.95

    def test_bbox_property(self):
        """Test bbox property."""
        face = FaceDetectionResult((10, 20, 50, 60))
        assert face.bbox == (10, 20, 50, 60)

    def test_center_property(self):
        """Test center property calculation."""
        face = FaceDetectionResult((10, 20, 50, 60))
        assert face.center == (35, 50)  # (10 + 50//2, 20 + 60//2)

    def test_repr(self):
        """Test string representation."""
        face = FaceDetectionResult((10, 20, 50, 60), 0.95)
        repr_str = repr(face)
        assert "Face(" in repr_str
        assert "x=10" in repr_str
        assert "conf=0.95" in repr_str


class TestDetectFaces:
    """Test main face detection function."""

    def test_detect_faces_with_file(self, sample_image_file):
        """Test face detection with image file."""
        faces = detect_faces(sample_image_file, method="haar")
        assert isinstance(faces, list)
        # Note: Haar cascade might not detect our simple synthetic face
        # This test mainly checks that the function runs without error

    def test_detect_faces_with_array(self, sample_image):
        """Test face detection with numpy array."""
        faces = detect_faces(sample_image, method="haar")
        assert isinstance(faces, list)

    def test_invalid_method(self, sample_image_file):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Method must be"):
            detect_faces(sample_image_file, method="invalid")

    def test_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            detect_faces("nonexistent.jpg")

    def test_return_image_option(self, sample_image_file):
        """Test return_image option."""
        result = detect_faces(
            sample_image_file, method="haar", return_image=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        faces, image = result
        assert isinstance(faces, list)
        assert isinstance(image, np.ndarray)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_overlap_no_overlap(self):
        """Test overlap calculation with no overlap."""
        face1 = FaceDetectionResult((0, 0, 10, 10))
        face2 = FaceDetectionResult((20, 20, 10, 10))
        overlap = _calculate_overlap(face1, face2)
        assert overlap == 0.0

    def test_calculate_overlap_partial(self):
        """Test overlap calculation with partial overlap."""
        face1 = FaceDetectionResult((0, 0, 20, 20))
        face2 = FaceDetectionResult((10, 10, 20, 20))
        overlap = _calculate_overlap(face1, face2)
        assert 0 < overlap < 1

    def test_calculate_overlap_complete(self):
        """Test overlap calculation with complete overlap."""
        face1 = FaceDetectionResult((0, 0, 20, 20))
        face2 = FaceDetectionResult((0, 0, 20, 20))
        overlap = _calculate_overlap(face1, face2)
        assert overlap == 1.0

    def test_remove_duplicate_faces_empty(self):
        """Test duplicate removal with empty list."""
        result = _remove_duplicate_faces([])
        assert result == []

    def test_remove_duplicate_faces_single(self):
        """Test duplicate removal with single face."""
        face = FaceDetectionResult((0, 0, 10, 10))
        result = _remove_duplicate_faces([face])
        assert len(result) == 1
        assert result[0] == face

    def test_remove_duplicate_faces_no_duplicates(self):
        """Test duplicate removal with no duplicates."""
        face1 = FaceDetectionResult((0, 0, 10, 10))
        face2 = FaceDetectionResult((50, 50, 10, 10))
        result = _remove_duplicate_faces([face1, face2])
        assert len(result) == 2

    def test_remove_duplicate_faces_with_duplicates(self):
        """Test duplicate removal with duplicates."""
        face1 = FaceDetectionResult((0, 0, 20, 20), 0.8)
        # Overlapping, higher confidence
        face2 = FaceDetectionResult((5, 5, 20, 20), 0.9)
        result = _remove_duplicate_faces([face1, face2])
        assert len(result) == 1
        assert result[0] == face2  # Should keep higher confidence face


# Integration tests would require actual images or more sophisticated synthetic images
# For now, we focus on unit tests of individual components
