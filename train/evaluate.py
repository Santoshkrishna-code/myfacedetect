"""Evaluation helper for detector models.

This script will attempt to run evaluation using ultralytics (if available)
or will provide guidance how to evaluate exported models.

Usage:
    python train/evaluate.py --weights runs/train/exp/weights/best.pt --data data/data.yaml
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to weights')
    p.add_argument('--data', required=True, help='Path to data.yaml')
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print('ultralytics not installed. Install with: pip install ultralytics')
        print('Alternatively, convert your model to ONNX and run COCO mAP evaluation tools')
        return

    model = YOLO(args.weights)
    print('Running val() — this will compute mAP on provided dataset')
    results = model.val(data=args.data)
    print(results.metrics)

if __name__ == '__main__':
    main()
