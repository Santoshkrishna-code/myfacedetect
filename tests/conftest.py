"""
Test configuration and fixtures for MyFaceDetect library.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_image():
    """Create a sample image with a simple face-like pattern."""
    # Create a 200x200 BGR image
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Draw a simple face-like pattern
    # Face outline (circle)
    cv2.circle(img, (100, 100), 80, (100, 100, 100), -1)

    # Eyes
    cv2.circle(img, (75, 75), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (125, 75), 10, (0, 0, 0), -1)  # Right eye

    # Nose
    cv2.circle(img, (100, 100), 5, (50, 50, 50), -1)

    # Mouth
    cv2.ellipse(img, (100, 130), (20, 10), 0, 0, 180, (0, 0, 0), 2)

    return img


@pytest.fixture
def sample_image_file(sample_image, tmp_path):
    """Save sample image to a temporary file."""
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), sample_image)
    return str(image_path)


@pytest.fixture
def empty_image():
    """Create an empty image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
