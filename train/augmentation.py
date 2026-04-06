"""Data augmentation utilities for face detection training.

Uses albumentations for efficient augmentation pipeline.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_train_augmentation(image_size: int = 640) -> Optional['A.Compose']:
    """Get augmentation pipeline for training.
    
    Args:
        image_size: Target image size for resizing
        
    Returns:
        albumentations.Compose object or None if albumentations not installed
    """
    if not HAS_ALBUMENTATIONS:
        print("Warning: albumentations not installed. Install with: pip install albumentations")
        return None
    
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.Affine(scale=(0.9, 1.1), p=0.3),
        
        # Photometric
        A.Brightness(limit=0.2, p=0.3),
        A.Contrast(limit=0.2, p=0.3),
        A.GaussNoise(p=0.1),
        A.GaussBlur(p=0.1),
        
        # Structural
        A.CoarseDropout(max_holes=2, max_height=20, max_width=20, p=0.1),
        
        # Resize to target size
        A.Resize(image_size, image_size),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))


def get_val_augmentation(image_size: int = 640) -> Optional['A.Compose']:
    """Get augmentation pipeline for validation (minimal).
    
    Args:
        image_size: Target image size for resizing
        
    Returns:
        albumentations.Compose object or None if albumentations not installed
    """
    if not HAS_ALBUMENTATIONS:
        return None
    
    return A.Compose([
        A.Resize(image_size, image_size),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))


def apply_augmentation(image: np.ndarray, bboxes: List[Dict], transform: Optional['A.Compose']) -> Tuple[np.ndarray, List[Dict]]:
    """Apply augmentation to image and bboxes.
    
    Args:
        image: Input image (cv2 format: BGR)
        bboxes: List of bounding boxes with keys 'x', 'y', 'w', 'h'
        transform: Augmentation transform from albumentations
        
    Returns:
        Tuple of (augmented_image, augmented_bboxes)
    """
    if transform is None:
        return image, bboxes
    
    # Convert bboxes to pascal_voc format (x_min, y_min, x_max, y_max)
    pascal_bboxes = []
    for bbox in bboxes:
        x_min = bbox['x']
        y_min = bbox['y']
        x_max = bbox['x'] + bbox['w']
        y_max = bbox['y'] + bbox['h']
        pascal_bboxes.append([x_min, y_min, x_max, y_max])
    
    # Apply transforms
    augmented = transform(image=image, bboxes=pascal_bboxes)
    aug_image = augmented['image']
    aug_bboxes_pascal = augmented['bboxes']
    
    # Convert back to our format (x, y, w, h)
    aug_bboxes = []
    for bbox_pascal in aug_bboxes_pascal:
        x_min, y_min, x_max, y_max = bbox_pascal
        aug_bboxes.append({
            'x': int(x_min),
            'y': int(y_min),
            'w': int(x_max - x_min),
            'h': int(y_max - y_min)
        })
    
    return aug_image, aug_bboxes


def create_mosaic_image(images: List[np.ndarray], size: int = 640) -> np.ndarray:
    """Create a mosaic image from 4 images (YOLOv4 style).
    
    Args:
        images: List of 4 images
        size: Output size
        
    Returns:
        Mosaic image of shape (size, size, 3)
    """
    import cv2
    
    if len(images) != 4:
        raise ValueError("Mosaic requires exactly 4 images")
    
    half = size // 2
    mosaic = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Resize and place images
    positions = [(0, 0), (half, 0), (0, half), (half, half)]
    for img, (y_off, x_off) in zip(images, positions):
        resized = cv2.resize(img, (half, half))
        mosaic[y_off:y_off+half, x_off:x_off+half] = resized
    
    return mosaic


def mixup(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Mix two images together (linear interpolation).
    
    Args:
        image1: First image
        image2: Second image  
        alpha: Mixing ratio (0.0 = all image2, 1.0 = all image1)
        
    Returns:
        Mixed image
    """
    # Ensure same size
    import cv2
    h, w = image1.shape[:2]
    image2_resized = cv2.resize(image2, (w, h))
    
    mixed = cv2.addWeighted(image1, alpha, image2_resized, 1 - alpha, 0)
    return mixed.astype(np.uint8)


if __name__ == '__main__':
    import os
    import cv2
    from data_loader import CustomImageDataset
    
    # Test augmentation
    dataset = CustomImageDataset('data/images', 'train')
    dataset.load()
    
    if len(dataset) > 0:
        train_aug = get_train_augmentation()
        image, boxes, _ = dataset[0]
        
        if train_aug:
            aug_image, aug_boxes = apply_augmentation(image, boxes, train_aug)
            print(f"Original: {image.shape}, boxes: {len(boxes)}")
            print(f"Augmented: {aug_image.shape}, boxes: {len(aug_boxes)}")
