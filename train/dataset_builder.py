"""Dataset builder for collecting and organizing face photos for training.

Features:
- Automatic face extraction from photos
- Duplicate detection and removal
- Quality filtering (brightness, sharpness, blur)
- Annotation management
- Dataset statistics
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib
from collections import defaultdict
import json


class PhotoQualityAnalyzer:
    """Analyze photo quality metrics."""
    
    @staticmethod
    def compute_sharpness(image: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Sharpness score (higher is sharper)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        return float(sharpness)
    
    @staticmethod
    def compute_brightness(image: np.ndarray) -> float:
        """Compute average brightness level.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Brightness score (0-255)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return float(brightness)
    
    @staticmethod
    def compute_contrast(image: np.ndarray) -> float:
        """Compute contrast (standard deviation of pixel values).
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Contrast score
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        return float(contrast)
    
    @staticmethod
    def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
        """Check if image is blurry.
        
        Args:
            image: Input image (BGR)
            threshold: Sharpness threshold below which image is considered blurry
            
        Returns:
            True if blurry, False otherwise
        """
        sharpness = PhotoQualityAnalyzer.compute_sharpness(image)
        return sharpness < threshold
    
    @staticmethod
    def is_too_dark(image: np.ndarray, threshold: float = 50.0) -> bool:
        """Check if image is too dark.
        
        Args:
            image: Input image (BGR)
            threshold: Brightness threshold
            
        Returns:
            True if too dark, False otherwise
        """
        brightness = PhotoQualityAnalyzer.compute_brightness(image)
        return brightness < threshold
    
    @staticmethod
    def is_too_bright(image: np.ndarray, threshold: float = 200.0) -> bool:
        """Check if image is too bright (overexposed).
        
        Args:
            image: Input image (BGR)
            threshold: Brightness threshold
            
        Returns:
            True if too bright, False otherwise
        """
        brightness = PhotoQualityAnalyzer.compute_brightness(image)
        return brightness > threshold
    
    @staticmethod
    def get_quality_score(image: np.ndarray) -> Dict[str, float]:
        """Get comprehensive quality metrics.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Dictionary with quality metrics
        """
        return {
            'sharpness': PhotoQualityAnalyzer.compute_sharpness(image),
            'brightness': PhotoQualityAnalyzer.compute_brightness(image),
            'contrast': PhotoQualityAnalyzer.compute_contrast(image),
        }


class DuplicateDetector:
    """Detect and remove duplicate images."""
    
    @staticmethod
    def compute_hash(image: np.ndarray, hash_size: int = 8) -> str:
        """Compute perceptual hash (pHash) of image.
        
        Args:
            image: Input image (BGR)
            hash_size: Hash grid size (8x8 default)
            
        Returns:
            Hex hash string
        """
        # Resize to hash_size x hash_size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size, hash_size))
        
        # Compute DCT
        dct = cv2.dct(np.float32(resized))
        
        # Average DCT values
        avg = np.mean(dct)
        
        # Create binary hash
        hash_bits = (dct > avg).flatten()
        hash_int = 0
        for i, bit in enumerate(hash_bits):
            if bit:
                hash_int |= (1 << i)
        
        return format(hash_int, '016x')
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Hamming distance
        """
        val = int(hash1, 16) ^ int(hash2, 16)
        return bin(val).count('1')
    
    @staticmethod
    def find_duplicates(image_paths: List[str], threshold: int = 10) -> List[List[str]]:
        """Find duplicate image groups.
        
        Args:
            image_paths: List of image file paths
            threshold: Hamming distance threshold for duplicates
            
        Returns:
            List of duplicate groups
        """
        hashes = {}
        duplicate_groups = []
        
        for path in image_paths:
            try:
                image = cv2.imread(path)
                if image is None:
                    continue
                
                hash_val = DuplicateDetector.compute_hash(image)
                
                # Check against existing hashes
                found_group = False
                for group in duplicate_groups:
                    ref_hash = hashes[group[0]]
                    distance = DuplicateDetector.hamming_distance(hash_val, ref_hash)
                    
                    if distance <= threshold:
                        group.append(path)
                        hashes[path] = hash_val
                        found_group = True
                        break
                
                if not found_group:
                    duplicate_groups.append([path])
                    hashes[path] = hash_val
            
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        # Return only actual duplicates (groups with >1 image)
        return [g for g in duplicate_groups if len(g) > 1]


class DatasetBuilder:
    """Build and organize face detection training datasets."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_analyzer = PhotoQualityAnalyzer()
        self.duplicate_detector = DuplicateDetector()
        self.metadata = defaultdict(dict)
    
    def import_photos(self, photo_dir: str, max_photos: Optional[int] = None) -> Dict:
        """Import photos from directory and apply quality filters.
        
        Args:
            photo_dir: Directory containing photos
            max_photos: Maximum number of photos to import
            
        Returns:
            Statistics dictionary
        """
        photo_dir = Path(photo_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Find all images
        image_paths = []
        for root, _, files in os.walk(photo_dir):
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, f))
        
        image_paths = sorted(image_paths)
        if max_photos:
            image_paths = image_paths[:max_photos]
        
        stats = {
            'total_found': len(image_paths),
            'imported': 0,
            'rejected': {
                'blurry': 0,
                'too_dark': 0,
                'too_bright': 0,
                'error': 0
            },
            'duplicates_removed': 0
        }
        
        print(f"Importing {len(image_paths)} photos...")
        
        # Process each image
        valid_paths = []
        for i, path in enumerate(image_paths):
            try:
                image = cv2.imread(path)
                if image is None:
                    stats['rejected']['error'] += 1
                    continue
                
                # Quality checks
                if self.quality_analyzer.is_blurry(image):
                    stats['rejected']['blurry'] += 1
                    continue
                
                if self.quality_analyzer.is_too_dark(image):
                    stats['rejected']['too_dark'] += 1
                    continue
                
                if self.quality_analyzer.is_too_bright(image):
                    stats['rejected']['too_bright'] += 1
                    continue
                
                valid_paths.append(path)
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                stats['rejected']['error'] += 1
        
        # Remove duplicates
        print(f"Checking for duplicates among {len(valid_paths)} valid images...")
        duplicates = self.duplicate_detector.find_duplicates(valid_paths)
        
        unique_paths = set(valid_paths)
        for dup_group in duplicates:
            # Keep first, remove rest
            for dup_path in dup_group[1:]:
                unique_paths.discard(dup_path)
                stats['duplicates_removed'] += 1
        
        unique_paths = list(unique_paths)
        
        # Copy and organize
        output_subdir = self.output_dir / 'photos'
        output_subdir.mkdir(exist_ok=True)
        
        for i, path in enumerate(unique_paths):
            try:
                image = cv2.imread(path)
                quality = self.quality_analyzer.get_quality_score(image)
                
                # Save with naming
                filename = f"photo_{i:06d}.jpg"
                output_path = output_subdir / filename
                cv2.imwrite(str(output_path), image)
                
                # Store metadata
                self.metadata[filename] = {
                    'source': str(path),
                    'quality': quality,
                    'timestamp': i
                }
                
                stats['imported'] += 1
                
            except Exception as e:
                print(f"Error saving {path}: {e}")
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(dict(self.metadata), f, indent=2)
        
        print(f"✅ Import complete: {stats['imported']} photos processed")
        print(f"   Duplicates removed: {stats['duplicates_removed']}")
        print(f"   Rejected (blurry): {stats['rejected']['blurry']}")
        print(f"   Rejected (too dark): {stats['rejected']['too_dark']}")
        print(f"   Rejected (too bright): {stats['rejected']['too_bright']}")
        
        return stats
    
    def extract_faces(self, detector=None) -> Dict:
        """Extract face regions from photos.
        
        Args:
            detector: Optional face detector object
            
        Returns:
            Statistics about extracted faces
        """
        if detector is None:
            # Use built-in detector
            try:
                import mediapipe as mp
                mp_face = mp.solutions.face_detection
                detector = mp_face.FaceDetection(min_detection_confidence=0.5)
            except ImportError:
                print("MediaPipe not available. Using OpenCV Haar cascade.")
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                detector = cascade
        
        photo_dir = self.output_dir / 'photos'
        faces_dir = self.output_dir / 'faces'
        faces_dir.mkdir(exist_ok=True)
        
        stats = {'total_photos': 0, 'faces_extracted': 0, 'error': 0}
        
        for i, photo_path in enumerate(sorted(photo_dir.glob('*.jpg'))):
            try:
                image = cv2.imread(str(photo_path))
                if image is None:
                    continue
                
                stats['total_photos'] += 1
                h, w = image.shape[:2]
                
                # Detect faces (simple version)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                # Extract and save faces
                for j, (x, y, fw, fh) in enumerate(faces):
                    # Add margin
                    margin = int(0.2 * fw)
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(w, x + fw + margin)
                    y2 = min(h, y + fh + margin)
                    
                    face_img = image[y1:y2, x1:x2]
                    
                    # Resize to standard size
                    face_img = cv2.resize(face_img, (224, 224))
                    
                    # Save
                    face_filename = f"face_{i:06d}_{j:02d}.jpg"
                    face_path = faces_dir / face_filename
                    cv2.imwrite(str(face_path), face_img)
                    
                    stats['faces_extracted'] += 1
            
            except Exception as e:
                print(f"Error extracting faces from {photo_path}: {e}")
                stats['error'] += 1
        
        print(f"✅ Face extraction complete: {stats['faces_extracted']} faces extracted from {stats['total_photos']} photos")
        
        return stats
    
    def generate_report(self) -> Dict:
        """Generate dataset report.
        
        Returns:
            Dictionary with dataset statistics
        """
        photo_dir = self.output_dir / 'photos'
        faces_dir = self.output_dir / 'faces'
        
        photo_count = len(list(photo_dir.glob('*.jpg'))) if photo_dir.exists() else 0
        face_count = len(list(faces_dir.glob('*.jpg'))) if faces_dir.exists() else 0
        
        # Compute quality stats
        quality_stats = {'sharpness': [], 'brightness': [], 'contrast': []}
        
        if self.metadata:
            for filename, meta in self.metadata.items():
                if 'quality' in meta:
                    quality_stats['sharpness'].append(meta['quality'].get('sharpness', 0))
                    quality_stats['brightness'].append(meta['quality'].get('brightness', 0))
                    quality_stats['contrast'].append(meta['quality'].get('contrast', 0))
        
        report = {
            'dataset_dir': str(self.output_dir),
            'total_photos': photo_count,
            'total_faces': face_count,
            'average_quality': {
                'sharpness': np.mean(quality_stats['sharpness']) if quality_stats['sharpness'] else 0,
                'brightness': np.mean(quality_stats['brightness']) if quality_stats['brightness'] else 0,
                'contrast': np.mean(quality_stats['contrast']) if quality_stats['contrast'] else 0,
            },
            'photo_dir': str(photo_dir),
            'faces_dir': str(faces_dir)
        }
        
        # Save report
        report_path = self.output_dir / 'dataset_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n📊 Dataset Report:")
        print(f"   Photos: {report['total_photos']}")
        print(f"   Extracted faces: {report['total_faces']}")
        print(f"   Avg Sharpness: {report['average_quality']['sharpness']:.2f}")
        print(f"   Avg Brightness: {report['average_quality']['brightness']:.2f}")
        print(f"   Avg Contrast: {report['average_quality']['contrast']:.2f}")
        
        return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build face detection datasets from photos')
    parser.add_argument('--input-dir', required=True, help='Directory containing photos')
    parser.add_argument('--output-dir', default='data/dataset', help='Output directory')
    parser.add_argument('--max-photos', type=int, help='Maximum number of photos to import')
    parser.add_argument('--extract-faces', action='store_true', help='Extract individual faces')
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(args.output_dir)
    stats = builder.import_photos(args.input_dir, args.max_photos)
    
    if args.extract_faces:
        face_stats = builder.extract_faces()
    
    report = builder.generate_report()
