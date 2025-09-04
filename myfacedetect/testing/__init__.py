"""
MyFaceDetect Testing Module
==========================

This module provides comprehensive testing and validation tools for the MyFaceDetect library.

Available Tools:
- FaceAugmentationTester: Test recognition robustness with various image augmentations
- FaceLabelingSystem: Demonstrate complete face recognition and labeling pipeline

Usage:
    from myfacedetect.testing import FaceAugmentationTester, FaceLabelingSystem
    
    # Create augmentation tester
    tester = FaceAugmentationTester(recognizer, detector)
    results = tester.test_recognition_robustness(image, "PersonName")
    
    # Create labeling system  
    labeling = FaceLabelingSystem(recognizer, detector)
    result = labeling.recognize_and_label_face(image)
"""

from .augmentation_test import FaceAugmentationTester
from .labeling_demo import FaceLabelingSystem

__all__ = ['FaceAugmentationTester', 'FaceLabelingSystem']
