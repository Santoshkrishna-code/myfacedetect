"""Prepare and verify datasets for training.

This is a small utility that checks a directory of images and annotations
and can prepare a YOLO/COCO-style manifest. It is a starting point and
should be adapted to your dataset.

Usage:
    python train/prepare_dataset.py --src path/to/raw_dataset --out data --format yolov8

Note: this script is intentionally lightweight and does not change images.
"""
import argparse
import os
import csv
import json
from pathlib import Path


def scan_images(src_dir):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                images.append(os.path.join(root, f))
    return sorted(images)


def write_simple_yaml(out_dir, train_list, val_list, names):
    data = {
        'train': str(train_list[0]) if len(train_list) == 1 else '\n'.join([str(p) for p in train_list]),
        'val': str(val_list[0]) if len(val_list) == 1 else '\n'.join([str(p) for p in val_list]),
        'nc': len(names),
        'names': names
    }
    with open(os.path.join(out_dir, 'data.yaml'), 'w', encoding='utf8') as fh:
        yaml_lines = [f"train: {train_list[0]}\n", f"val: {val_list[0]}\n", f"nc: {len(names)}\n", f"names: {names}\n"]
        fh.writelines(yaml_lines)
    print('Wrote data.yaml')


def create_simple_manifest(images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'manifest.csv')
    with open(csv_path, 'w', newline='', encoding='utf8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['image_path'])
        for p in images:
            writer.writerow([p])
    print('Wrote', csv_path)
    return csv_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Source image folder')
    p.add_argument('--out', default='data', help='Output folder for prepared dataset')
    p.add_argument('--format', choices=['yolov8', 'coco', 'simple'], default='simple')
    p.add_argument('--val-split', type=float, default=0.2)
    args = p.parse_args()

    images = scan_images(args.src)
    if not images:
        print('No images found in', args.src)
        return

    os.makedirs(args.out, exist_ok=True)

    n_val = max(1, int(len(images) * args.val_split))
    val = images[:n_val]
    train = images[n_val:]

    print(f'Found {len(images)} images — train: {len(train)}, val: {len(val)}')

    if args.format == 'simple':
        create_simple_manifest(train + val, args.out)
    elif args.format == 'yolov8':
        # Minimal data.yaml for ultralytics training; adapt as needed
        names = ['face']
        write_simple_yaml(args.out, train, val, names)
    elif args.format == 'coco':
        print('COCO conversion not implemented — add conversion steps here')


if __name__ == '__main__':
    main()
