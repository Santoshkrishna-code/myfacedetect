# Face Detection Training Module

A production-ready training framework for face detection models with advanced features for hyperparameter optimization, experiment tracking, and model management.

## Overview

This training module provides a complete pipeline for training face detection models with:
- **Modular architecture** for easy customization
- **Advanced training techniques** (mixed precision, gradient accumulation)
- **Experiment management** for tracking and comparing runs
- **Hyperparameter optimization** using Bayesian search
- **Comprehensive testing** and validation

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_training.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### 2. Basic Training

```bash
# Run training with default config
python training_pipeline.py --config config_template.yaml

# Run with custom experiment name
python training_pipeline.py --config config_template.yaml --exp-name "my_experiment" --device cuda
```

### 3. Review Results

```bash
# Check experiment results
python -c "from experiment_tracking import ExperimentTracker; tracker = ExperimentTracker(); print([e['name'] for e in tracker.list_experiments()])"
```

## Module Structure

```
train/
├── config_management.py           # Configuration handling
├── custom_trainer.py              # Advanced PyTorch trainer
├── experiment_tracking.py         # Experiment management
├── hyperparameter_tuning.py       # Bayesian optimization
├── training_pipeline.py           # Main orchestrator
├── training_utils.py              # Utility functions
├── training_examples.py           # Usage examples
├── test_training_pipeline.py      # Tests
├── config_template.yaml           # Example configuration
├── requirements_training.txt      # Dependencies
├── TRAINING_README.md             # Detailed documentation
└── README.md                       # This file
```

## Key Features

### 1. Configuration Management

Define all training parameters in YAML:

```yaml
data:
  dataset_path: ./data
  batch_size: 32
  img_size: 416

model:
  name: yolov8
  num_classes: 2

training:
  num_epochs: 100
  learning_rate: 0.001
  optimizer: adam
  device: cuda
```

**Features**:
- Multiple format support (YAML, JSON)
- Environment variable interpolation
- Configuration validation
- Automatic type conversion

### 2. Custom Trainer

Production-ready training loop with:

```python
from custom_trainer import CustomTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    use_mixed_precision=True,
    early_stopping_patience=10,
    device='cuda'
)

# Create trainer
trainer = CustomTrainer(model, config)

# Train
history = trainer.fit(train_loader, val_loader)
```

**Features**:
- Mixed precision training (automatic loss scaling)
- Gradient accumulation for larger effective batches
- Learning rate scheduling (cosine, linear, step)
- Early stopping with configurable patience
- Automatic checkpoint management

### 3. Experiment Tracking

Track and compare multiple experiments:

```python
from experiment_tracking import ExperimentTracker

tracker = ExperimentTracker()

# Create experiment
exp_id = tracker.create_experiment(
    'baseline',
    config={'lr': 1e-3, 'batch_size': 32},
    'Baseline model'
)

# Log metrics
tracker.start_experiment(exp_id)
for epoch in range(100):
    tracker.log_metrics(exp_id, epoch, loss=loss, accuracy=acc)

tracker.complete_experiment(exp_id, {'mAP': 0.95})

# Compare experiments
best = tracker.get_best_experiment('mAP', mode='max')
```

**Features**:
- Centralized experiment metadata
- Per-epoch metric logging
- Experiment comparison and ranking
- Results export (CSV, JSON)

### 4. Hyperparameter Optimization

Optimize hyperparameters using Bayesian search:

```python
from hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(study_name='face_detection')

def training_func(params, trial):
    # Train with params and return metric
    return mAP

# Run optimization
tuner.optimize(training_func, n_trials=100, n_jobs=4)

# Get best parameters
best_params = tuner.get_best_params()
```

**Features**:
- Bayesian optimization with TPE sampler
- Multi-objective optimization support
- Parallel trial execution
- Automatic visualization of results

### 5. Training Pipeline

Orchestrates complete training workflow:

```python
from training_pipeline import TrainingPipeline
from config_management import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config('config.yaml')

# Create and run pipeline
pipeline = TrainingPipeline(config)
results = pipeline.run()
```

**Workflow**:
1. Setup experiment tracking
2. Prepare data loaders
3. Create model
4. Train with custom trainer
5. Evaluate on test set
6. Save results and checkpoints

## Usage Examples

### Example 1: Basic Training

```bash
cd train/
python training_pipeline.py --config config_template.yaml
```

### Example 2: Hyperparameter Tuning

```python
from training_examples import example_hyperparameter_tuning
example_hyperparameter_tuning()
```

### Example 3: Experiment Comparison

```python
from experiment_tracking import ExperimentTracker
import pandas as pd

tracker = ExperimentTracker()
exps = tracker.list_experiments(status='completed')
df = tracker.compare_experiments([e['id'] for e in exps])
print(df)
```

### Example 4: Custom Metrics

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

pipeline = TrainingPipeline(config)
results = pipeline.run(metric_fn=compute_metrics)
```

## Configuration Guide

### Data Configuration

```yaml
data:
  dataset_path: /path/to/dataset      # Required: path to dataset
  train_split: 0.8                    # Training/total ratio
  batch_size: 32                      # Batch size for training
  num_workers: 4                      # Parallel data loading workers
  img_size: 416                       # Input image size
  augmentation: true                  # Enable augmentation
```

### Model Configuration

```yaml
model:
  name: yolov8                        # Model architecture
  backbone: resnet50                  # Backbone network
  pretrained: true                    # Use pretrained weights
  num_classes: 2                      # Number of classes (e.g., face/no-face)
  dropout: 0.1                        # Dropout rate
```

### Training Configuration

```yaml
training:
  num_epochs: 100                     # Total epochs
  learning_rate: 0.001               # Initial learning rate
  optimizer: adam                     # 'adam', 'sgd', 'adamw'
  scheduler: cosine                   # 'cosine', 'linear', 'step'
  warmup_epochs: 5                    # Warmup period
  mixed_precision: true               # Use AMP
  gradient_accumulation_steps: 1      # Accumulation steps
  early_stopping_patience: 10         # Early stopping patience
  device: cuda                        # 'cuda' or 'cpu'
```

## Advanced Features

### Mixed Precision Training

Automatically enabled for faster training on supported GPUs:

```yaml
training:
  mixed_precision: true
```

Benefits:
- 2-3x speedup on modern GPUs
- Reduced memory usage
- Maintained accuracy

### Gradient Accumulation

Use for larger effective batch sizes:

```yaml
training:
  gradient_accumulation_steps: 4
  batch_size: 8  # Effective batch size: 32
```

### Learning Rate Scheduling

Multiple scheduling strategies:

```yaml
training:
  scheduler: cosine          # Best for most cases
  # or
  scheduler: linear          # Linear decay
  # or
  scheduler: step            # Step decay
  lr_decay_steps: 30
  lr_decay_rate: 0.1
```

### Early Stopping

Prevent overfitting:

```yaml
training:
  early_stopping_patience: 10        # Stop after 10 epochs without improvement
  early_stopping_min_delta: 0.0001   # Minimum improvement threshold
```

## Performance Tips

### 1. Data Loading
```yaml
data:
  batch_size: 64                # Larger batches
  num_workers: 8                # More parallel workers
  pin_memory: true              # Pin to GPU memory
```

### 2. Training Optimization
```yaml
training:
  mixed_precision: true         # Enable AMP
  gradient_accumulation_steps: 2
  learning_rate: 0.001          # Appropriate learning rate
  warmup_epochs: 5              # Gradual warmup
```

### 3. Model Architecture
- Use pretrained backbones when possible
- Balance model size and accuracy
- Consider inference time requirements

## Troubleshooting

### Out of Memory (OOM)
```yaml
# Reduce batch size
data:
  batch_size: 16  # Decrease from 32

# Enable gradient accumulation
training:
  gradient_accumulation_steps: 4

# Enable mixed precision
training:
  mixed_precision: true
```

### Slow Training
```yaml
# Increase data loading workers
data:
  num_workers: 8
  pin_memory: true

# Enable mixed precision
training:
  mixed_precision: true

# Use larger batch size (if possible)
data:
  batch_size: 64
```

### Validation Accuracy Not Improving
```yaml
# Reduce learning rate
training:
  learning_rate: 0.0005

# Increase warmup
training:
  warmup_epochs: 10

# Try different scheduler
training:
  scheduler: linear

# Check data pipeline for issues
```

## Testing

Run comprehensive tests:

```bash
# Run all tests
pytest test_training_pipeline.py -v

# Run specific test class
pytest test_training_pipeline.py::TestCustomTrainer -v

# Run with coverage
pytest test_training_pipeline.py --cov=. --cov-report=html
```

## Output Files

After training, check the following directories:

```
experiments/
├── metadata.json                    # All experiments metadata
└── {exp_id}/
    ├── config.json                  # Training configuration
    ├── results.json                 # Final results and metrics
    └── metrics/                     # Per-epoch metrics

checkpoints/
└── {exp_id}/
    ├── best_model.pt                # Best validation checkpoint
    ├── checkpoint_epoch_*.pt        # Periodic checkpoints
    └── model_final.pt               # Final model

logs/
├── training_*.log                   # Training logs
└── tensorboard/                     # TensorBoard events
```

## API Reference

### ConfigManager

```python
manager = ConfigManager()
config = manager.load_config('config.yaml')
manager.save_config(config, 'output.json', format='json')
merged = manager.merge_configs(config, {'training': {'num_epochs': 200}})
```

### CustomTrainer

```python
trainer = CustomTrainer(model, config)
history = trainer.fit(train_loader, val_loader)
trainer.save_checkpoint('model.pt')
trainer.load_checkpoint('model.pt')
```

### ExperimentTracker

```python
tracker = ExperimentTracker()
exp_id = tracker.create_experiment('name', config)
tracker.log_metrics(exp_id, epoch, loss=0.5)
tracker.complete_experiment(exp_id, results)
tracker.list_experiments()
tracker.compare_experiments([exp_id_1, exp_id_2])
```

### HyperparameterTuner

```python
tuner = HyperparameterTuner()
tuner.optimize(training_func, n_trials=100)
best_params = tuner.get_best_params()
tuner.save_results('results.json')
tuner.visualize_results('plots/')
```

## Contributing

To extend the training module:

1. Add custom augmentation in data loading code
2. Implement custom model architectures
3. Add new metrics in metric computation functions
4. Extend configuration schema for new parameters

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check troubleshooting section
2. Review examples in `training_examples.py`
3. Check test cases in `test_training_pipeline.py`
4. Refer to detailed documentation in `TRAINING_README.md`
