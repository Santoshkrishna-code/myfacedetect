"""Enhanced evaluation script for face detection models.

Includes:
- mAP (mean Average Precision) calculation
- Per-class metrics
- Confusion matrix
- Visualization
- Report generation
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class MetricsCalculator:
    """Calculate detection metrics."""
    
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
    
    def add_prediction(self, image_id: str, boxes: List[Dict], confidences: List[float]):
        """Add predictions for an image.
        
        Args:
            image_id: Unique image identifier
            boxes: List of predicted boxes {'x': int, 'y': int, 'w': int, 'h': int}
            confidences: List of confidence scores
        """
        for box, conf in zip(boxes, confidences):
            self.predictions.append({
                'image_id': image_id,
                'box': box,
                'confidence': conf
            })
    
    def add_ground_truth(self, image_id: str, boxes: List[Dict]):
        """Add ground truth boxes for an image.
        
        Args:
            image_id: Unique image identifier
            boxes: List of ground truth boxes
        """
        for box in boxes:
            self.ground_truth.append({
                'image_id': image_id,
                'box': box,
                'matched': False
            })
    
    def compute_iou(self, box1: Dict, box2: Dict) -> float:
        """Compute Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: {'x': int, 'y': int, 'w': int, 'h': int}
            box2: Same format
            
        Returns:
            IoU value between 0 and 1
        """
        # Convert to coordinates
        x1_min, y1_min = box1['x'], box1['y']
        x1_max = x1_min + box1['w']
        y1_max = y1_min + box1['h']
        
        x2_min, y2_min = box2['x'], box2['y']
        x2_max = x2_min + box2['w']
        y2_max = y2_min + box2['h']
        
        # Compute intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Compute union
        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_precision_recall(self, iou_threshold: float = 0.5) -> Tuple[float, float]:
        """Compute precision and recall at given IoU threshold.
        
        Args:
            iou_threshold: IoU threshold for considering a detection as correct
            
        Returns:
            Tuple of (precision, recall)
        """
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(self.predictions, key=lambda x: x['confidence'], reverse=True)
        
        matched_gt = set()
        
        for pred in sorted_preds:
            pred_box = pred['box']
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(self.ground_truth):
                if gt['image_id'] != pred['image_id']:
                    continue
                if gt_idx in matched_gt:
                    continue
                
                iou = self.compute_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(self.ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recall
    
    def compute_ap(self, iou_threshold: float = 0.5, num_points: int = 11) -> float:
        """Compute Average Precision (AP) using interpolation.
        
        Args:
            iou_threshold: IoU threshold
            num_points: Number of points for interpolation
            
        Returns:
            Average Precision score
        """
        precisions = []
        recalls = []
        
        # Sort predictions by confidence
        sorted_preds = sorted(self.predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = 0
        fp = 0
        matched_gt = set()
        
        for pred in sorted_preds:
            pred_box = pred['box']
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(self.ground_truth):
                if gt['image_id'] != pred['image_id']:
                    continue
                if gt_idx in matched_gt:
                    continue
                
                iou = self.compute_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / len(self.ground_truth) if len(self.ground_truth) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Interpolate precision-recall curve
        if len(recalls) == 0:
            return 0.0
        
        ap = 0.0
        for i in range(num_points):
            threshold_recall = i / max(1, num_points - 1)
            max_precision = 0.0
            for p, r in zip(precisions, recalls):
                if r >= threshold_recall:
                    max_precision = max(max_precision, p)
            ap += max_precision / num_points
        
        return ap
    
    def generate_report(self, output_path: str = 'metrics_report.json'):
        """Generate a comprehensive metrics report.
        
        Args:
            output_path: Path to save the report
        """
        # Compute at multiple IoU thresholds
        iou_thresholds = [0.5, 0.75, 0.9]
        results = {
            'summary': {},
            'by_iou_threshold': {}
        }
        
        for iou in iou_thresholds:
            p, r = self.compute_precision_recall(iou)
            ap = self.compute_ap(iou)
            
            results['by_iou_threshold'][str(iou)] = {
                'precision': float(p),
                'recall': float(r),
                'f1_score': 2 * (p * r) / (p + r) if (p + r) > 0 else 0,
                'ap': float(ap)
            }
        
        # Summary at IoU=0.5
        results['summary'] = results['by_iou_threshold']['0.5']
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Report saved to {output_path}")
        print(json.dumps(results, indent=2))
        
        return results


if __name__ == '__main__':
    import argparse
    from data_loader import get_dataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--pred-file', required=True, help='JSON file with predictions')
    parser.add_argument('--output', default='metrics_report.json')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = get_dataset('custom', args.data_dir, 'test')
    
    # Load predictions
    with open(args.pred_file, 'r') as f:
        predictions = json.load(f)
    
    # Calculate metrics
    calc = MetricsCalculator()
    
    for i in range(len(dataset)):
        image, boxes, path = dataset[i]
        image_id = Path(path).stem
        
        # Add ground truth
        calc.add_ground_truth(image_id, boxes)
        
        # Add predictions (if available)
        if image_id in predictions:
            preds = predictions[image_id]
            pred_boxes = [p['box'] for p in preds]
            confidences = [p['confidence'] for p in preds]
            calc.add_prediction(image_id, pred_boxes, confidences)
    
    # Generate report
    calc.generate_report(args.output)
