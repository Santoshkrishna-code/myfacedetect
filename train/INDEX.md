# Face Detection Training Module - File Index

## Core Training Modules

### 1. `config_management.py`
**Purpose**: Configuration handling and validation
**Key Classes**:
- `Config`: Main configuration dataclass
- `DataConfig`: Data-specific configuration
- `ModelConfig`: Model architecture configuration
- `TrainingConfig`: Training parameters
- `AugmentationConfig`: Data augmentation parameters
- `ConfigManager`: Load, save, and merge configurations

**Usage**:
```python
from config_management import ConfigManager
manager = ConfigManager()
config = manager.load_config('config.yaml')
```

---

### 2. `custom_trainer.py`
**Purpose**: Advanced PyTorch training loop
**Key Classes**:
- `CustomTrainer`: Main trainer implementing training loop
- `TrainingConfig`: Configuration for trainer
- `MetricsTracker`: Track metrics per batch and epoch

**Features**:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Automatic checkpointing

**Usage**:
```python
from custom_trainer import CustomTrainer, TrainingConfig
trainer = CustomTrainer(model, config)
history = trainer.fit(train_loader, val_loader)
```

---

### 3. `experiment_tracking.py`
**Purpose**: Experiment management and comparison
**Key Classes**:
- `ExperimentTracker`: Main tracker for experiments
- `CheckpointManager`: Manage model checkpoints

**Features**:
- Create and track experiments
- Log per-epoch metrics
- Compare multiple experiments
- Export results to CSV
- Automatic checkpoint management

**Usage**:
```python
from experiment_tracking import ExperimentTracker
tracker = ExperimentTracker()
exp_id = tracker.create_experiment('baseline', config)
tracker.log_metrics(exp_id, epoch, loss=0.5)
```

---

### 4. `hyperparameter_tuning.py`
**Purpose**: Hyperparameter optimization using Bayesian search
**Key Classes**:
- `HyperparameterTuner`: Main tuning orchestrator using Optuna
- `TrialCallback`: Monitor trials during optimization

**Features**:
- Bayesian optimization (TPE sampler)
- Multi-objective optimization
- Parallel trial execution
- Result visualization
- Best model extraction

**Usage**:
```python
from hyperparameter_tuning import HyperparameterTuner
tuner = HyperparameterTuner()
tuner.optimize(training_func, n_trials=100, n_jobs=4)
best_params = tuner.get_best_params()
```

---

### 5. `training_pipeline.py`
**Purpose**: Main training orchestrator
**Key Classes**:
- `TrainingPipeline`: Orchestrates complete training workflow

**Workflow**:
1. Setup experiment tracking
2. Prepare data loaders
3. Create model
4. Train with custom trainer
5. Evaluate on test set
6. Save results and checkpoints

**Usage**:
```python
from training_pipeline import TrainingPipeline
pipeline = TrainingPipeline(config)
results = pipeline.run()
```

---

### 6. `training_utils.py`
**Purpose**: Utility functions for training
**Functions**:
- Device utilities: `get_device()`, `get_device_stats()`, `clear_cache()`
- Data utilities: `compute_normalization_stats()`, `normalize_image()`, `denormalize_image()`
- Statistics: `compute_statistics()`, `compute_moving_average()`
- File utilities: `compute_file_hash()`, `compute_dict_hash()`
- Checkpoint utilities: `get_checkpoint_size()`, `estimate_model_size()`
- Visualization: `plot_training_history()`, `plot_metric_distribution()`
- Logging: `setup_logging()`

**Usage**:
```python
from training_utils import get_device, estimate_model_size
device = get_device()
model_size = estimate_model_size(model)
```

---

## Example and Testing Files

### 7. `training_examples.py`
**Purpose**: Comprehensive usage examples
**Examples**:
1. Basic training with CustomTrainer
2. Experiment tracking workflow
3. Hyperparameter tuning
4. Custom metrics computation
5. Complete training pipeline

**Usage**:
```bash
python training_examples.py --example 1
python training_examples.py --example 0  # Run all
```

---

### 8. `test_training_pipeline.py`
**Purpose**: Comprehensive test suite
**Test Classes**:
- `TestConfigManagement`: Configuration tests
- `TestMetricsTracker`: Metrics tracking tests
- `TestExperimentTracker`: Experiment tracking tests
- `TestCheckpointManager`: Checkpoint management tests
- `TestHyperparameterTuner`: Hyperparameter tuning tests
- `TestCustomTrainer`: Trainer tests
- `TestIntegration`: Integration tests
- `TestUtilities`: Utility function tests

**Usage**:
```bash
pytest test_training_pipeline.py -v
pytest test_training_pipeline.py::TestCustomTrainer -v
```

---

## Configuration and Documentation

### 9. `config_template.yaml`
**Purpose**: Example configuration file
**Sections**:
- `data`: Data loading parameters
- `model`: Model architecture
- `training`: Training parameters
- `augmentation`: Data augmentation
- Experiment settings

**Usage**:
```bash
python training_pipeline.py --config config_template.yaml
```

---

### 10. `requirements_training.txt`
**Purpose**: Python dependencies
**Contents**:
- Core: torch, torchvision, numpy, pandas
- Config: PyYAML, jsonschema, python-dotenv
- Tuning: optuna, plotly
- Tracking: wandb, tensorboard
- Utilities: scipy, scikit-learn, matplotlib, seaborn
- Development: pytest, black, flake8, mypy

**Installation**:
```bash
pip install -r requirements_training.txt
```

---

## Documentation Files

### 11. `README.md`
**Purpose**: Main documentation
**Sections**:
- Quick start
- Module structure
- Key features
- Configuration guide
- Advanced features
- Performance tips
- Troubleshooting
- API reference

---

### 12. `TRAINING_README.md`
**Purpose**: Detailed training documentation
**Sections**:
- Feature overview
- Component descriptions
- Installation guide
- Usage examples
- Configuration schema
- Output structure
- Performance tips
- Advanced examples
- Troubleshooting

---

### 13. `REFERENCE.md`
**Purpose**: Complete reference guide
**Sections**:
- Overview
- Files created
- System architecture
- Feature comparison
- Statistics
- Usage patterns
- Extension guide
- Performance characteristics
- Testing coverage
- Best practices

---

## Quick Navigation

### By Use Case

**Just want to train?**
1. Start with `config_template.yaml`
2. Run `training_pipeline.py`
3. Check results with `ExperimentTracker`

**Want to tune hyperparameters?**
1. Use `hyperparameter_tuning.py`
2. Define training function
3. Run `tuner.optimize()`

**Need to understand how it works?**
1. Read `README.md`
2. Check examples in `training_examples.py`
3. Review `TRAINING_README.md`

**Want to extend functionality?**
1. Subclass from core modules
2. Check tests in `test_training_pipeline.py`
3. Refer to API in `TRAINING_README.md`

### By Component

**Configuration**: `config_management.py` + `config_template.yaml`
**Training**: `custom_trainer.py` + `training_pipeline.py`
**Tracking**: `experiment_tracking.py`
**Optimization**: `hyperparameter_tuning.py`
**Utilities**: `training_utils.py`
**Testing**: `test_training_pipeline.py`
**Documentation**: `README.md`, `TRAINING_README.md`, `REFERENCE.md`

### By Experience Level

**Beginner**:
1. `README.md` - Quick start
2. `training_examples.py --example 0` - Run examples
3. Modify `config_template.yaml`

**Intermediate**:
1. `TRAINING_README.md` - Detailed guide
2. Review source code in modules
3. Run custom training with `CustomTrainer`

**Advanced**:
1. Study architecture in `REFERENCE.md`
2. Extend modules with custom functionality
3. Use `test_training_pipeline.py` as examples

---

## Module Interdependencies

```
External Libraries
    ├── torch, torchvision (deep learning)
    ├── optuna (hyperparameter tuning)
    ├── pandas (data manipulation)
    └── numpy (numerical computing)

config_management.py
    └── Used by: training_pipeline.py

custom_trainer.py
    ├── Uses: training_utils.py
    └── Used by: training_pipeline.py

experiment_tracking.py
    └── Used by: training_pipeline.py

hyperparameter_tuning.py
    └── Standalone module

training_pipeline.py
    ├── Uses: config_management.py
    ├── Uses: custom_trainer.py
    ├── Uses: experiment_tracking.py
    ├── Uses: training_utils.py
    └── Main orchestrator

training_utils.py
    └── Used by: All modules

training_examples.py
    ├── Examples for: All modules
    └── Demonstrates: Complete workflows

test_training_pipeline.py
    └── Tests for: All modules
```

---

## Common Commands

```bash
# Installation
pip install -r requirements_training.txt

# Training
python training_pipeline.py --config config_template.yaml

# Examples
python training_examples.py --example 1
python training_examples.py --example 0

# Testing
pytest test_training_pipeline.py -v
pytest test_training_pipeline.py -v --cov=.

# View experiments
python -c "from experiment_tracking import ExperimentTracker; \
tracker = ExperimentTracker(); \
for e in tracker.list_experiments(): print(e['name'])"

# Compare experiments
python -c "from experiment_tracking import ExperimentTracker; \
tracker = ExperimentTracker(); \
exps = [e['id'] for e in tracker.list_experiments()]; \
print(tracker.compare_experiments(exps))"
```

---

## File Sizes and Metrics

| File | Lines | Classes | Functions | Purpose |
|------|-------|---------|-----------|---------|
| config_management.py | 500 | 6 | 20+ | Configuration |
| custom_trainer.py | 600 | 3 | 25+ | Training |
| experiment_tracking.py | 500 | 2 | 30+ | Tracking |
| hyperparameter_tuning.py | 400 | 2 | 15+ | Optimization |
| training_pipeline.py | 500 | 1 | 20+ | Orchestration |
| training_utils.py | 500 | 0 | 50+ | Utilities |
| training_examples.py | 600 | 5 | 10+ | Examples |
| test_training_pipeline.py | 600 | 10 | 30+ | Tests |
| **Total** | **4200+** | **30+** | **200+** | **Production-ready** |

---

## Getting Started

1. **Read**: `README.md` (10 minutes)
2. **Install**: `pip install -r requirements_training.txt` (2 minutes)
3. **Run Example**: `python training_examples.py --example 1` (5-10 minutes)
4. **Customize**: Edit `config_template.yaml` (10 minutes)
5. **Train**: `python training_pipeline.py --config config.yaml` (Variable)

---

## Support Resources

- **Quick Questions**: See troubleshooting in `README.md`
- **Detailed Help**: Read `TRAINING_README.md`
- **Architecture**: Review `REFERENCE.md`
- **Code Examples**: Check `training_examples.py`
- **Testing**: Run `test_training_pipeline.py`

---

## Version Information

- PyTorch: >= 2.0.0
- Python: >= 3.9
- Optuna: >= 3.0.0
- CUDA: Optional (works with CPU)

---

**Last Updated**: Created as comprehensive training framework
**Status**: Production-ready with full documentation and testing
