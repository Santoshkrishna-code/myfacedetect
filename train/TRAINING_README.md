# Advanced Training Pipeline

A comprehensive training pipeline for deep learning models with advanced features including hyperparameter tuning, experiment tracking, mixed precision training, and checkpoint management.

## Features

### Core Features
- **Modular Architecture**: Separate components for configuration, training, evaluation, and tracking
- **Hyperparameter Optimization**: Bayesian optimization using Optuna
- **Experiment Tracking**: Track experiments, compare results, and manage checkpoints
- **Custom Training Loop**: Advanced PyTorch trainer with:
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Learning rate scheduling
  - Early stopping
  - Checkpointing

### Advanced Features
- **Configuration Management**:
  - YAML/JSON configuration files
  - Environment variable interpolation
  - Configuration validation
  - Schema-based validation

- **Metrics Tracking**:
  - Per-batch and per-epoch metrics
  - Automatic metric averaging
  - Custom metric functions

- **Checkpoint Management**:
  - Save/load checkpoints
  - Best model tracking
  - Automatic cleanup of old checkpoints

## Components

### 1. Configuration Management (`config_management.py`)
Handles all configuration-related tasks:
- Load configurations from YAML/JSON
- Validate configurations against schema
- Interpolate environment variables
- Support for multiple configuration dataclasses

**Key Classes**:
- `Config`: Main configuration dataclass
- `ConfigManager`: Manager for loading/saving configurations

### 2. Custom Trainer (`custom_trainer.py`)
PyTorch-based training loop with advanced features:
- Mixed precision training
- Gradient accumulation
- Automatic checkpoint management
- Early stopping
- Learning rate scheduling

**Key Classes**:
- `CustomTrainer`: Main trainer class
- `TrainingConfig`: Configuration for trainer
- `MetricsTracker`: Track metrics during training

### 3. Experiment Tracking (`experiment_tracking.py`)
Track and manage training experiments:
- Create and track experiments
- Log metrics per epoch
- Compare multiple experiments
- Export results to CSV

**Key Classes**:
- `ExperimentTracker`: Main experiment tracker
- `CheckpointManager`: Manage model checkpoints

### 4. Hyperparameter Tuning (`hyperparameter_tuning.py`)
Optimize hyperparameters using Bayesian optimization:
- Define search spaces
- Run parallel trials
- Visualize optimization results
- Extract best parameters

**Key Classes**:
- `HyperparameterTuner`: Main tuning class
- `TrialCallback`: Callback for trial monitoring

### 5. Training Pipeline (`training_pipeline.py`)
Orchestrates the complete training workflow:
- Coordinate all components
- Manage data loading
- Handle model creation
- Run training and evaluation

**Key Classes**:
- `TrainingPipeline`: Main pipeline orchestrator

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements_training.txt
```

2. **Verify installation**:
```bash
python -c "import torch; print(torch.__version__)"
```

## Usage

### Basic Training

1. **Create configuration file** (`config.yaml`):
```yaml
data:
  dataset_path: ./data/coco
  batch_size: 32
  num_workers: 4
  img_size: 416

model:
  name: yolov8
  backbone: resnet50
  num_classes: 80

training:
  num_epochs: 100
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine
  device: cuda
```

2. **Run training**:
```bash
python training_pipeline.py --config config.yaml --exp-name "baseline"
```

### Advanced Usage

#### Custom Trainer with Mixed Precision
```python
from custom_trainer import CustomTrainer, TrainingConfig

config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    use_mixed_precision=True,
    device='cuda'
)

trainer = CustomTrainer(model, config)
history = trainer.fit(train_loader, val_loader)
```

#### Hyperparameter Tuning
```python
from hyperparameter_tuning import HyperparameterTuner

def training_func(params, trial):
    # Train model with parameters and return metric
    return metric_value

tuner = HyperparameterTuner(study_name='face_detection')
tuner.optimize(training_func, n_trials=100, n_jobs=4)

best_params = tuner.get_best_params()
results = tuner.save_results('tuning_results.json')
```

#### Experiment Tracking
```python
from experiment_tracking import ExperimentTracker

tracker = ExperimentTracker()

# Create experiment
exp_id = tracker.create_experiment(
    'yolov8_baseline',
    config={'batch_size': 32, 'lr': 0.001},
    'Baseline YOLO v8 detector'
)

# Log metrics
tracker.start_experiment(exp_id)
for epoch in range(100):
    tracker.log_metrics(exp_id, epoch, loss=loss_value, accuracy=acc_value)

tracker.complete_experiment(exp_id, {'final_mAP': 0.95})

# Compare experiments
best = tracker.get_best_experiment('final_mAP', mode='max')
```

#### Configuration Management
```python
from config_management import ConfigManager

manager = ConfigManager()

# Load configuration
config = manager.load_config('config.yaml')

# Modify and merge
override = {'training': {'num_epochs': 200}}
merged_config = manager.merge_configs(config, override)

# Save configuration
manager.save_config(merged_config, 'new_config.yaml', format='yaml')
```

## Configuration Schema

### Data Configuration
```yaml
data:
  dataset_path: string (required)
  train_split: float (0.0-1.0, default: 0.8)
  val_split: float (0.0-1.0, default: 0.1)
  test_split: float (0.0-1.0, default: 0.1)
  batch_size: int (default: 32)
  num_workers: int (default: 4)
  img_size: int (default: 416)
```

### Model Configuration
```yaml
model:
  name: string (required)
  backbone: string (resnet50, etc.)
  pretrained: bool (default: true)
  num_classes: int (default: 80)
  dropout: float (default: 0.1)
```

### Training Configuration
```yaml
training:
  num_epochs: int (default: 100)
  learning_rate: float (default: 0.001)
  optimizer: 'adam' | 'sgd' | 'adamw'
  scheduler: 'cosine' | 'linear' | 'step'
  device: 'cuda' | 'cpu'
  mixed_precision: bool (default: true)
  early_stopping_patience: int (default: 10)
```

## Output Structure

After training, the following directory structure is created:

```
experiments/
├── metadata.json                          # All experiments metadata
└── {exp_name}_{timestamp}/
    ├── config.json                        # Configuration used
    ├── results.json                       # Final results
    └── metrics.json                       # Per-epoch metrics

checkpoints/
└── {exp_name}_{timestamp}/
    ├── best_model.pt                      # Best model checkpoint
    ├── checkpoint_epoch_5.pt
    ├── checkpoint_epoch_10.pt
    └── model_final.pt                     # Final model

logs/
├── tensorboard events
└── training logs
```

## Performance Tips

1. **Mixed Precision Training**: Enable for faster training on supported GPUs
   ```yaml
   training:
     mixed_precision: true
   ```

2. **Gradient Accumulation**: Use for effective larger batch sizes
   ```yaml
   training:
     gradient_accumulation_steps: 4
   ```

3. **Parallel Data Loading**: Increase workers for faster loading
   ```yaml
   data:
     num_workers: 8
     pin_memory: true
   ```

4. **Learning Rate Scheduling**: Use cosine annealing for best results
   ```yaml
   training:
     scheduler: cosine
     warmup_epochs: 5
   ```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in configuration
- Enable `gradient_accumulation_steps`
- Use `mixed_precision: true`

### Slow Training
- Increase `num_workers` for data loading
- Enable `pin_memory: true`
- Use `mixed_precision: true` on supported GPUs

### Validation Accuracy Not Improving
- Increase `warmup_epochs`
- Try different `learning_rate`
- Check data pipeline for issues
- Increase `num_epochs`

## Advanced Examples

### Custom Metric Function
```python
def compute_metrics(predictions, labels):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

pipeline = TrainingPipeline(config)
results = pipeline.run(metric_fn=compute_metrics)
```

### Resume Training from Checkpoint
```python
trainer = CustomTrainer(model, config)
trainer.load_checkpoint('checkpoints/exp_name/best_model.pt')
history = trainer.fit(train_loader, val_loader)
```

### Compare Experiments
```python
tracker = ExperimentTracker()

# Get experiments
exps = tracker.list_experiments(status='completed')
exp_ids = [e['id'] for e in exps[-5:]]  # Last 5 experiments

# Compare
comparison_df = tracker.compare_experiments(exp_ids)
print(comparison_df)

# Export results
tracker.export_results('comparison.csv')
```

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Automatic Mixed Precision](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

## License

This project is licensed under the MIT License - see LICENSE file for details.
