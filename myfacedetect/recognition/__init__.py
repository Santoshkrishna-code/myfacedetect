"""
Recognition Module
Advanced face recognition capabilities with database management.
"""

from .face_recognition import FaceRecognizer, create_face_recognizer
from .database import FaceDatabase, create_face_database

__all__ = [
    'FaceRecognizer',
    'create_face_recognizer',
    'FaceDatabase',
    'create_face_database'
]
