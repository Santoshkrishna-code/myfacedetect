"""Data loading and preprocessing utilities for face detection training.

Supports:
- WIDER FACE format
- COCO format
- Custom image folders with annotations
- Automatic train/val/test splits
"""
import os
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


class FaceDataset:
    """Base class for face detection datasets."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.images = []
        self.annotations = []
    
    def load(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        raise NotImplementedError


class WIDERFaceDataset(FaceDataset):
    """Load WIDER FACE dataset format."""
    
    def load(self):
        split_dir = self.data_dir / 'WIDER_' + self.split / 'images'
        annotation_file = self.data_dir / f'wider_face_split' / f'wider_face_{self.split}_bbx_gt.txt'
        
        if not split_dir.exists():
            raise FileNotFoundError(f"WIDER FACE split directory not found: {split_dir}")
        
        self.images = []
        self.annotations = []
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            img_path = lines[i].strip()
            i += 1
            num_faces = int(lines[i].strip())
            i += 1
            
            boxes = []
            for _ in range(num_faces):
                box_data = lines[i].strip().split()
                x, y, w, h = map(int, box_data[:4])
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
                i += 1
            
            full_img_path = str(split_dir / img_path)
            if os.path.exists(full_img_path):
                self.images.append(full_img_path)
                self.annotations.append(boxes)
        
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        boxes = self.annotations[idx]
        return image, boxes, img_path


class CustomImageDataset(FaceDataset):
    """Load custom image folder with optional CSV annotations."""
    
    def load(self, annotation_file: Optional[str] = None):
        self.images = []
        self.annotations = []
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    self.images.append(os.path.join(root, f))
        
        # Load annotations if provided
        if annotation_file and os.path.exists(annotation_file):
            annotations_dict = {}
            with open(annotation_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_name = row['image']
                    boxes = json.loads(row['boxes'])
                    annotations_dict[img_name] = boxes
            self.annotations = [annotations_dict.get(Path(img).name, []) for img in self.images]
        else:
            self.annotations = [[] for _ in self.images]
        
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        boxes = self.annotations[idx]
        return image, boxes, img_path


def create_dataset_split(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15,
                        seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Create train/val/test splits from a directory of images."""
    np.random.seed(seed)
    
    images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                images.append(os.path.join(root, f))
    
    images = sorted(images)
    n = len(images)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_images = [images[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    test_images = [images[i] for i in test_idx]
    
    return train_images, val_images, test_images


def get_dataset(dataset_type: str, data_dir: str, split: str = 'train') -> FaceDataset:
    """Factory function to get the appropriate dataset."""
    if dataset_type.lower() == 'wider':
        dataset = WIDERFaceDataset(data_dir, split)
    elif dataset_type.lower() == 'custom':
        dataset = CustomImageDataset(data_dir, split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataset.load()
    return dataset


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to dataset directory')
    parser.add_argument('--type', choices=['wider', 'custom'], default='custom')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()
    
    dataset = get_dataset(args.type, args.data_dir, args.split)
    print(f"Loaded {len(dataset)} images")
    
    if len(dataset) > 0:
        image, boxes, path = dataset[0]
        print(f"Sample image: {path}")
        print(f"Shape: {image.shape}")
        print(f"Boxes: {boxes}")
