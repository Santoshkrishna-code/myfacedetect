"""Minimal training wrapper for YOLOv8 (ultralytics) if available.

This script tries to import `ultralytics` and will print instructions
if the package is not installed. It's intended as a reproducible starter.

Usage:
    python train/train_detector.py --data data/data.yaml --model yolov8n.pt --epochs 50

Note: Training large models on CPU is slow. Use GPU where possible.
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to data.yaml')
    p.add_argument('--model', default='yolov8n.pt', help='Pretrained model or config')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print('ultralytics not installed. Install with: pip install ultralytics')
        print('Or run training with your preferred framework (PyTorch/TensorFlow)')
        return

    print('Starting YOLOv8 training:')
    print('data=', args.data, 'model=', args.model, 'epochs=', args.epochs)

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == '__main__':
    main()
