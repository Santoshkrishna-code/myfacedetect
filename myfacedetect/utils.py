"""
Utility functions for MyFaceDetect library.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
import json
import logging
from datetime import datetime

from .core import FaceDetectionResult

logger = logging.getLogger(__name__)


def benchmark_methods(image_paths: List[Union[str, Path]],
                      methods: List[str] = ["haar", "mediapipe"],
                      iterations: int = 3) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different detection methods on a set of images.

    Args:
        image_paths: List of image paths to test
        methods: List of methods to benchmark
        iterations: Number of iterations per method

    Returns:
        Dictionary with benchmark results
    """
    import time
    from .core import detect_faces

    results = {}

    for method in methods:
        method_results = {
            "total_time": 0,
            "average_time": 0,
            "total_faces": 0,
            "images_processed": 0,
            "errors": 0
        }

        logger.info(f"Benchmarking {method} method...")

        for iteration in range(iterations):
            for image_path in image_paths:
                try:
                    start_time = time.time()
                    faces = detect_faces(image_path, method=method)
                    end_time = time.time()

                    method_results["total_time"] += (end_time - start_time)
                    method_results["total_faces"] += len(faces)
                    method_results["images_processed"] += 1

                except Exception as e:
                    method_results["errors"] += 1
                    logger.error(
                        f"Error processing {image_path} with {method}: {e}")

        if method_results["images_processed"] > 0:
            method_results["average_time"] = (
                method_results["total_time"] /
                method_results["images_processed"]
            )
            method_results["faces_per_image"] = (
                method_results["total_faces"] /
                method_results["images_processed"]
            )

        results[method] = method_results

    return results


def create_face_dataset(source_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        method: str = "mediapipe",
                        min_face_size: int = 64,
                        max_faces_per_image: int = 10) -> Dict[str, int]:
    """
    Create a dataset of face crops from a directory of images.

    Args:
        source_dir: Directory containing source images
        output_dir: Directory to save face crops
        method: Detection method to use
        min_face_size: Minimum face size (width/height) to save
        max_faces_per_image: Maximum faces to extract per image

    Returns:
        Statistics about the dataset creation
    """
    from .core import detect_faces

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    stats = {
        "images_processed": 0,
        "faces_found": 0,
        "faces_saved": 0,
        "errors": 0
    }

    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in source_path.rglob('*')
                   if f.suffix.lower() in supported_formats]

    logger.info(f"Found {len(image_files)} images to process")

    for image_file in image_files:
        try:
            faces = detect_faces(image_file, method=method)
            stats["images_processed"] += 1
            stats["faces_found"] += len(faces)

            # Load image for cropping
            img = cv2.imread(str(image_file))
            if img is None:
                continue

            # Save face crops
            saved_count = 0
            for i, face in enumerate(faces[:max_faces_per_image]):
                x, y, w, h = face.bbox

                # Check minimum size
                if w < min_face_size or h < min_face_size:
                    continue

                # Extract face crop with some padding
                padding = 0.1
                pad_w = int(w * padding)
                pad_h = int(h * padding)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img.shape[1], x + w + pad_w)
                y2 = min(img.shape[0], y + h + pad_h)

                face_crop = img[y1:y2, x1:x2]

                # Save crop
                crop_filename = f"{image_file.stem}_face_{i+1:02d}.jpg"
                crop_path = output_path / crop_filename

                if cv2.imwrite(str(crop_path), face_crop):
                    saved_count += 1
                    stats["faces_saved"] += 1

            if saved_count > 0:
                logger.info(
                    f"Saved {saved_count} faces from {image_file.name}")

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Error processing {image_file}: {e}")

    # Save dataset statistics
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset creation complete: {stats}")
    return stats


def analyze_image_quality(image_path: Union[str, Path]) -> Dict[str, float]:
    """
    Analyze image quality metrics that might affect face detection.

    Args:
        image_path: Path to the image

    Returns:
        Dictionary with quality metrics
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    metrics = {}

    # Image resolution
    metrics["width"] = img.shape[1]
    metrics["height"] = img.shape[0]
    metrics["total_pixels"] = metrics["width"] * metrics["height"]

    # Brightness statistics
    metrics["mean_brightness"] = float(np.mean(gray))
    metrics["std_brightness"] = float(np.std(gray))

    # Contrast using Michelson contrast
    max_val = np.max(gray)
    min_val = np.min(gray)
    if max_val + min_val > 0:
        metrics["michelson_contrast"] = (
            max_val - min_val) / (max_val + min_val)
    else:
        metrics["michelson_contrast"] = 0.0

    # Sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics["sharpness"] = float(laplacian.var())

    # Noise estimation using local variance
    kernel = np.ones((3, 3), np.float32) / 9
    mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    noise_estimate = np.mean(np.abs(gray.astype(np.float32) - mean_filtered))
    metrics["noise_estimate"] = float(noise_estimate)

    return metrics


def create_detection_report(results: List[FaceDetectionResult],
                            image_path: Union[str, Path],
                            method: str,
                            execution_time: float) -> Dict[str, Any]:
    """
    Create a detailed report of face detection results.

    Args:
        results: List of detection results
        image_path: Path to the processed image
        method: Detection method used
        execution_time: Time taken for detection

    Returns:
        Detailed report dictionary
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    report = {
        "image_info": {
            "path": str(image_path),
            "filename": Path(image_path).name,
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": img.shape[2] if len(img.shape) == 3 else 1
        },
        "detection_info": {
            "method": method,
            "execution_time": execution_time,
            "faces_detected": len(results),
            "timestamp": datetime.now().isoformat()
        },
        "faces": []
    }

    # Add quality metrics
    report["quality_metrics"] = analyze_image_quality(image_path)

    # Add face details
    for i, face in enumerate(results):
        face_info = {
            "id": i + 1,
            "bbox": face.bbox,
            "center": face.center,
            "width": face.width,
            "height": face.height,
            "area": face.width * face.height,
            "aspect_ratio": face.width / face.height if face.height > 0 else 0
        }

        if hasattr(face, 'confidence'):
            face_info["confidence"] = face.confidence

        # Calculate relative position in image
        face_info["relative_position"] = {
            "x": face.x / img.shape[1],
            "y": face.y / img.shape[0],
            "width": face.width / img.shape[1],
            "height": face.height / img.shape[0]
        }

        report["faces"].append(face_info)

    return report


def visualize_detection_results(image_path: Union[str, Path],
                                faces: List[FaceDetectionResult],
                                method: str,
                                save_path: Optional[Union[str, Path]] = None,
                                show_confidence: bool = True,
                                show_id: bool = True) -> np.ndarray:
    """
    Create a visualization of face detection results.

    Args:
        image_path: Path to the original image
        faces: List of detected faces
        method: Detection method used
        save_path: Optional path to save the visualization
        show_confidence: Whether to show confidence scores
        show_id: Whether to show face IDs

    Returns:
        Annotated image as numpy array
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Create a copy for annotation
    annotated = img.copy()

    # Color scheme based on method
    colors = {
        "haar": (255, 0, 0),      # Blue
        "mediapipe": (0, 255, 0),  # Green
        "both": (0, 255, 255)     # Yellow
    }
    color = colors.get(method, (255, 255, 255))

    # Draw faces
    for i, face in enumerate(faces):
        x, y, w, h = face.bbox

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

        # Draw face ID
        if show_id:
            cv2.putText(annotated, f'#{i+1}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw confidence
        if show_confidence and hasattr(face, 'confidence') and face.confidence < 1.0:
            conf_text = f'{face.confidence:.2f}'
            cv2.putText(annotated, conf_text, (x + w - 80, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw center point
        center = face.center
        cv2.circle(annotated, center, 5, color, -1)

    # Add header information
    header_height = 60
    header = np.zeros((header_height, annotated.shape[1], 3), dtype=np.uint8)

    # Method and count
    method_text = f"Method: {method.upper()} | Faces: {len(faces)}"
    cv2.putText(header, method_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Image info
    img_text = f"Image: {Path(image_path).name} | Size: {img.shape[1]}x{img.shape[0]}"
    cv2.putText(header, img_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Combine header and image
    result = np.vstack([header, annotated])

    # Save if requested
    if save_path:
        cv2.imwrite(str(save_path), result)
        logger.info(f"Visualization saved to {save_path}")

    return result


def export_results(results: List[Dict[str, Any]],
                   output_file: Union[str, Path],
                   format: str = "json") -> None:
    """
    Export detection results to various formats.

    Args:
        results: List of detection result dictionaries
        output_file: Output file path
        format: Export format ("json", "csv", "xml")
    """
    output_path = Path(output_file)

    if format.lower() == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    elif format.lower() == "csv":
        import csv

        if not results:
            return

        # Flatten the results for CSV
        flattened_results = []
        for result in results:
            base_info = {
                "image_path": result["image_info"]["path"],
                "method": result["detection_info"]["method"],
                "execution_time": result["detection_info"]["execution_time"],
                "faces_detected": result["detection_info"]["faces_detected"]
            }

            if result["faces"]:
                for face in result["faces"]:
                    row = {**base_info}
                    row.update({
                        "face_id": face["id"],
                        "bbox_x": face["bbox"][0],
                        "bbox_y": face["bbox"][1],
                        "bbox_w": face["bbox"][2],
                        "bbox_h": face["bbox"][3],
                        "confidence": face.get("confidence", 1.0)
                    })
                    flattened_results.append(row)
            else:
                flattened_results.append(base_info)

        with open(output_path, 'w', newline='') as f:
            if flattened_results:
                writer = csv.DictWriter(
                    f, fieldnames=flattened_results[0].keys())
                writer.writeheader()
                writer.writerows(flattened_results)

    else:
        raise ValueError(f"Unsupported export format: {format}")

    logger.info(f"Results exported to {output_path} in {format} format")
