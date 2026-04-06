# Complete MyFaceDetect Training & Deployment Guide

This guide covers the complete workflow for training, evaluating, optimizing, and deploying face detection models using MyFaceDetect v0.3.0+.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Model Optimization](#model-optimization)
6. [Deployment](#deployment)
7. [Web UI](#web-ui)
8. [Advanced Topics](#advanced-topics)

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Santoshkrishna-code/myfacedetect.git
cd myfacedetect

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core package
pip install -e .

# Install training dependencies (optional)
pip install -r requirements-dev.txt

# For full features
pip install -e ".[training,ui,gpu]"
```

### 2. Run Live Detection

```bash
# CPU-only (using OpenCV Haar cascade fallback)
python examples/detect_faces_live.py

# Using batch script (Windows)
scripts\run_live_windows.bat

# Using Streamlit UI
streamlit run app.py
```

## Data Preparation

### Download WIDER FACE Dataset

```bash
# Create data directory
mkdir -p data/wider_face

# Download WIDER FACE (google drive or official source)
# Extract to data/wider_face/
```

### Prepare Custom Dataset

```bash
# Create directory structure
mkdir -p data/custom_images/train
mkdir -p data/custom_images/val
mkdir -p data/custom_images/test

# Copy images
cp /path/to/your/images/* data/custom_images/train/

# Create automatic train/val/test split
python train/data_loader.py \
    --data-dir data/custom_images \
    --type custom \
    --split train
```

### Create Annotations (CSV format)

```csv
image,boxes
img001.jpg,"[{""x"": 10, ""y"": 20, ""w"": 100, ""h"": 120}]"
img002.jpg,"[{""x"": 50, ""y"": 60, ""w"": 80, ""h"": 90}]"
```

### Using Annotation Tool

```bash
# For WIDER FACE format:
python train/prepare_dataset.py \
    --src data/wider_face \
    --out data \
    --format yolov8
```

## Training

### Install Training Dependencies

```bash
pip install -r requirements-dev.txt
# Or
pip install torch torchvision ultralytics albumentations scikit-learn pandas
```

### Train YOLOv8 Model

```bash
# Prepare data
python train/prepare_dataset.py \
    --src data/custom_images \
    --out data/prepared \
    --format yolov8

# Train detector
python train/train_detector.py \
    --data data/prepared/data.yaml \
    --model yolov8n.pt \  # nano, small, medium, large, xlarge
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0  # GPU index or 'cpu'

# Results saved in runs/train/exp/
```

### Custom Training Loop

```python
from train.train_detector import YOLOTrainer
from train.augmentation import get_train_augmentation, get_val_augmentation

# Create trainer
trainer = YOLOTrainer(
    model='yolov8n.pt',
    data_yaml='data/data.yaml',
    epochs=100,
    imgsz=640,
    batch_size=16,
    device='0'  # GPU or 'cpu'
)

# Train with callbacks
history = trainer.train(
    resume=False,  # Resume from checkpoint
    callbacks=[]   # Optional callbacks
)
```

### Data Augmentation

```python
from train.augmentation import get_train_augmentation, apply_augmentation
import cv2

# Load image and boxes
image = cv2.imread('image.jpg')
boxes = [{'x': 10, 'y': 20, 'w': 100, 'h': 120}]

# Get augmentation pipeline
augmentation = get_train_augmentation(image_size=640)

# Apply augmentation
aug_image, aug_boxes = apply_augmentation(image, boxes, augmentation)

print(f"Original: {image.shape}, boxes: {len(boxes)}")
print(f"Augmented: {aug_image.shape}, boxes: {len(aug_boxes)}")
```

## Evaluation

### Evaluate on Test Set

```bash
# Evaluate trained model
python train/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/data.yaml \
    --imgsz 640

# Outputs mAP@0.5, mAP@0.5:0.95, precision, recall
```

### Compute Detailed Metrics

```bash
# Run predictions on test set (generates predictions.json)
python train/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/data.yaml \
    --save-predictions

# Compute detailed metrics
python train/metrics.py \
    --data-dir data/custom_images/test \
    --pred-file predictions.json \
    --output metrics_report.json

# View report
cat metrics_report.json
```

### Analyze Results

```python
import json

with open('metrics_report.json') as f:
    metrics = json.load(f)

print("Metrics Summary:")
for iou_threshold, results in metrics['by_iou_threshold'].items():
    print(f"IoU={iou_threshold}:")
    print(f"  - Precision: {results['precision']:.4f}")
    print(f"  - Recall: {results['recall']:.4f}")
    print(f"  - F1-Score: {results['f1_score']:.4f}")
    print(f"  - AP: {results['ap']:.4f}")
```

## Model Optimization

### Convert to ONNX

```bash
# Convert PyTorch model to ONNX
python train/optimize.py \
    --action convert \
    --model-path runs/train/exp/weights/best.pt \
    --output-path models/face_detector.onnx \
    --input-shape 1,3,640,640
```

### Quantize Model (INT8)

```bash
# Quantize for faster inference
python train/optimize.py \
    --action quantize \
    --model-path models/face_detector.onnx \
    --output-path models/face_detector_quant.onnx

# File size reduction: ~4x smaller
```

### Benchmark Model

```bash
# Profile inference speed
python train/optimize.py \
    --action benchmark \
    --model-path models/face_detector.onnx \
    --input-shape 1,3,640,640

# Output: FPS, latency, memory usage
```

### Generate Model Report

```bash
# Create model specification report
python train/optimize.py \
    --action report \
    --model-path models/face_detector.onnx

# Outputs: model_report.json with specs
```

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t myfacedetect:latest .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data myfacedetect:latest

# Access at http://localhost:8501
```

### Docker Compose

```bash
# Start entire stack (web UI + API + database)
docker-compose up -d

# View logs
docker-compose logs -f web

# Stop services
docker-compose down
```

### Local Deployment

```bash
# Run Streamlit web UI
streamlit run app.py --server.maxUploadSize 200 --server.maxMessageSize 200

# Access at http://localhost:8501
```

## Web UI

### Features

1. **Image Upload**: Upload and analyze single images
2. **Batch Processing**: Process multiple images at once
3. **Webcam**: Real-time detection from webcam (local only)
4. **Configuration**: Adjust detection methods and parameters
5. **Export**: Download results as JSON/CSV

### Usage

```bash
# Start UI
streamlit run app.py

# Open browser: http://localhost:8501

# Upload image or use webcam
# Results shown in real-time with statistics
```

### Custom Styling

Edit `app.py` to customize:
- Detection methods
- Confidence thresholds
- Output format
- Branding and colors

## Advanced Topics

### Hyperparameter Tuning

```bash
# Optuna-based hyperparameter search
python train/hyperparameter_tuning.py \
    --data data/data.yaml \
    --n-trials 100 \
    --output trial_results.json
```

### Experiment Tracking

```bash
# Use Weights & Biases for experiment tracking
pip install wandb

# Login
wandb login

# Training with tracking
python train/train_detector.py \
    --data data/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --use-wandb  # Automatic logging
```

### Model Ensemble

```python
from myfacedetect import DetectorFactory

# Load multiple models
detector1 = DetectorFactory.create_detector('yolov8', config)
detector2 = DetectorFactory.create_detector('mediapipe', config)
detector3 = DetectorFactory.create_detector('retinaface', config)

# Ensemble detection
detections1 = detector1.detect(image)
detections2 = detector2.detect(image)
detections3 = detector3.detect(image)

# Combine results (voting)
combined = ensemble_detections([detections1, detections2, detections3])
```

### Real-time Inference

```python
import cv2
from myfacedetect import create_detector

# Load model in half-precision (faster on GPU)
detector = create_detector('yolov8n', half=True)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    faces = detector.detect(frame)
    
    # Draw results
    for face in faces:
        x, y, w, h = face['bbox']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train/train_detector.py --batch 8

# Use smaller model
python train/train_detector.py --model yolov8n.pt

# Enable gradient accumulation (2x effective batch size)
python train/train_detector.py --accumulate 2
```

### Slow Inference

```bash
# Use quantized model
python train/optimize.py --action quantize --model-path model.onnx

# Convert to half-precision
model = load_model('model.pt', half=True)

# Use smaller input size
detector.detect(image, imgsz=416)
```

### Low Accuracy

1. **Check data quality**: Verify annotations are correct
2. **Increase training data**: Collect more diverse samples
3. **Data augmentation**: Use stronger augmentation
4. **Train longer**: Increase epochs (100→200)
5. **Use larger model**: yolov8m or yolov8l
6. **Transfer learning**: Start from pretrained weights

## Best Practices

1. **Always use train/val/test splits**: No identity leakage
2. **Monitor metrics**: Track mAP, precision, recall
3. **Save checkpoints**: Save best weights during training
4. **Version models**: Keep track of model versions
5. **Document experiments**: Log hyperparameters and results
6. **Test before deployment**: Validate on held-out test set

## Performance Benchmarks (Reference)

| Hardware | Model | Input Size | FPS (CPU) | FPS (GPU) |
|----------|-------|-----------|----------|----------|
| Intel i7-10700K | YOLOv8n | 640 | 5-8 | 80-100 |
| Intel i7-10700K | YOLOv8n | 416 | 10-15 | 150-200 |
| Intel i5 (laptop) | YOLOv8n | 416 | 2-3 | N/A |
| NVIDIA RTX 3080 | YOLOv8m | 640 | - | 200-300 |
| NVIDIA RTX 3080 | YOLOv8n | 640 | - | 500+ |

## References

- [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Format](https://onnx.ai/)
- [Original MyFaceDetect README](../README.md)

---

For questions or issues, please open an issue on [GitHub](https://github.com/Santoshkrishna-code/myfacedetect/issues).
