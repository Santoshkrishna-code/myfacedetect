# Model Improvement & Training Plan

Goal: increase detection and recognition accuracy for `myfacedetect` by establishing a reproducible training and evaluation workflow, improving data quality, running experiments, and integrating model selection and deployment best practices.

This document gives an actionable roadmap and quick-start commands to run experiments on CPU/GPU systems.

## 1. Priorities (high → low)

- Data quality and coverage (collect diverse faces: age, ethnicity, lighting, occlusions)
- Clear train/val/test splits (no identity overlap between splits for recognition)
- Data augmentation and synthetic variants (Occlusion, blur, noise, color jitter)
- Transfer learning with modern detectors (YOLOv8, RetinaFace) for detection
- Embedding/recognition improvements with ArcFace (insightface)
- Proper metrics and evaluation pipeline (mAP, precision/recall, ROC/FAR/FRR)
- Hyperparameter tuning and model selection (random search, Optuna)
- Model optimization for inference (quantization, pruning, ONNX)
- Monitoring and reproducibility (experiment logs, checkpoints, seed control)

## 2. Datasets (suggested)

- Detection:
  - WIDER FACE (large, varied)
  - FDDB
  - Wider augmentation of custom photos
- Recognition:
  - VGGFace2 (for pretraining / transfer)
  - MS1M (careful with licensing)
  - Build a small labeled dataset of your users/people of interest

## 3. Recommended tools & libraries

- Training: PyTorch, ultralytics (YOLOv8), insightface (ArcFace) for recognition
- Augmentation: Albumentations
- Evaluation: COCO tools (mAP), scikit-learn (ROC/AUC)
- Experiment tracking: Weights & Biases, MLFlow, or plain TensorBoard
- Deployment: ONNX + onnxruntime (CPU friendly), TorchScript

## 4. Evaluation metrics

- Detection: mAP@0.5, mAP@[0.5:0.95], precision, recall, per-size APs
- Recognition: verification ROC curve, TAR@FAR, top-1/top-5 accuracy
- Latency: FPS at target resolution (CPU baseline)
- Model size: parameters / file size

## 5. Workflow & reproducibility

- Use a `data/` folder with a canonical structure (train/val/test)
- Store dataset manifests (CSV/YAML) with file paths and labels
- Fix random seeds in training scripts
- Save checkpoints and evaluation logs per-run
- Use version control for config files (YAML) and record model git commit

## 6. Quick experiment recipe (detection using YOLOv8)

1. Convert your dataset to YOLOv8/COCO format (see `train/prepare_dataset.py`).
2. Create `data.yaml` describing `train`, `val`, `nc` and `names`.
3. Run training (example using `train/train_detector.py`):

```bash
# (Prefer GPU) Example:
python train/train_detector.py --data data/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```

4. Evaluate model:

```bash
python train/evaluate.py --weights runs/train/exp/weights/best.pt --data data/data.yaml
```

5. Tune hyperparameters and repeat.

## 7. Recognition recipe (ArcFace / InsightFace)

- Use a face detector to crop aligned faces into `data/recognition/train` and `val` with folders per-identity.
- Train an ArcFace model using insightface training utilities or a simple PyTorch head on top of a pre-trained backbone.

## 8. Data augmentation suggestions

- Random crop, scale, rotation (small angles)
- Color jitter, brightness/contrast
- Gaussian blur, motion blur
- Occlusion patches (synthetic occluders)
- Random horizontal flip (where applicable)
- MixUp and CutMix (for robustness)

## 9. Model optimization for CPU

- Use small/lightweight backbones (yolov8n, MobileNet, EfficientNet-lite)
- Quantize post-training INT8 if accuracy drop acceptable
- Export to ONNX and benchmark with `onnxruntime` CPU
- Consider knowledge distillation from a larger teacher model

## 10. Experiment logging & tracking

- Log metrics per epoch, and per-evaluation run
- Save artifacts (weights, model config, evaluation CSV)
- Use unique run IDs (timestamp + git commit)

## 11. Next steps (concrete tasks to implement)

- [ ] Add `requirements-dev.txt` with training packages
- [ ] Add `train/prepare_dataset.py` to convert/verify datasets
- [ ] Add `train/train_detector.py` minimal trainer using ultralytics (fallback instructions)
- [ ] Add `train/evaluate.py` to compute mAP and basic metrics
- [ ] Start with small experiment on a subset of WIDER FACE to validate end-to-end
- [ ] Add experiment tracking (WandB/TensorBoard)

---

See `train/` for starter scripts. Run `python train/prepare_dataset.py --help` for usage. Adapt and extend each script to your preferred models and datasets.
