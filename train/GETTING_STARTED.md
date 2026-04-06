# Getting Started Checklist

Complete this checklist to get up and running with the advanced training framework.

## ✅ Installation Phase (5 minutes)

- [ ] **Step 1**: Navigate to train directory
  ```bash
  cd d:\myfacedetect\train
  ```

- [ ] **Step 2**: Install Python dependencies
  ```bash
  pip install -r requirements_training.txt
  ```

- [ ] **Step 3**: Verify PyTorch installation
  ```bash
  python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
  ```

- [ ] **Step 4**: Verify Optuna installation
  ```bash
  python -c "import optuna; print(f'Optuna {optuna.__version__}')"
  ```

## 📚 Learning Phase (15 minutes)

- [ ] **Step 5**: Read quick start guide
  - Open `README.md`
  - Read "Quick Start" section

- [ ] **Step 6**: Review architecture
  - Open `REFERENCE.md`
  - Review "System Architecture" section

- [ ] **Step 7**: Check examples
  - Review `training_examples.py`
  - Note the 5 different examples

## 🏃 Running Examples (5-10 minutes)

- [ ] **Step 8**: Run all examples
  ```bash
  python training_examples.py --example 0
  ```

- [ ] **Step 9**: Run specific example
  ```bash
  python training_examples.py --example 1  # Basic training
  ```

- [ ] **Step 10**: Check experiment tracker
  ```bash
  python -c "from experiment_tracking import ExperimentTracker; \
  tracker = ExperimentTracker(); \
  for e in tracker.list_experiments(): print(f'{e[\"name\"]}: {e[\"status\"]}')"
  ```

## ⚙️ Configuration Phase (10 minutes)

- [ ] **Step 11**: Review configuration template
  - Open `config_template.yaml`
  - Understand YAML structure

- [ ] **Step 12**: Create custom configuration
  ```bash
  cp config_template.yaml my_config.yaml
  ```

- [ ] **Step 13**: Edit configuration
  - Update `dataset_path` to your dataset
  - Adjust `batch_size` based on GPU memory
  - Set `num_epochs` as needed

- [ ] **Step 14**: Validate configuration
  ```python
  from config_management import ConfigManager
  manager = ConfigManager()
  config = manager.load_config('my_config.yaml')
  print("Configuration loaded successfully!")
  ```

## 🎯 Running Training (Variable)

- [ ] **Step 15**: Start basic training
  ```bash
  python training_pipeline.py --config my_config.yaml
  ```

- [ ] **Step 16**: Monitor training progress
  - Watch console output
  - Check loss decreasing

- [ ] **Step 17**: Training completed
  - Model saved in `checkpoints/`
  - Results saved in `experiments/`

## 🔍 Analysis Phase (10 minutes)

- [ ] **Step 18**: View experiment results
  ```python
  from experiment_tracking import ExperimentTracker
  tracker = ExperimentTracker()
  exp = tracker.get_best_experiment('loss', mode='min')
  print(f"Best experiment: {exp['id']}")
  print(f"Results: {exp['results']}")
  ```

- [ ] **Step 19**: Compare experiments
  ```python
  exps = tracker.list_experiments(status='completed')
  df = tracker.compare_experiments([e['id'] for e in exps])
  print(df)
  ```

- [ ] **Step 20**: Export results
  ```python
  tracker.export_results('results.csv')
  ```

## 🧪 Testing Phase (5 minutes)

- [ ] **Step 21**: Run all tests
  ```bash
  pytest test_training_pipeline.py -v
  ```

- [ ] **Step 22**: Check test coverage
  ```bash
  pytest test_training_pipeline.py --cov=. --cov-report=html
  ```

- [ ] **Step 23**: All tests passing
  - Review test output
  - Confirm 20+ tests passed

## 🚀 Advanced Usage (Optional)

- [ ] **Step 24**: Hyperparameter tuning
  ```bash
  python training_examples.py --example 3
  ```

- [ ] **Step 25**: Custom metrics
  ```bash
  python training_examples.py --example 4
  ```

- [ ] **Step 26**: Extend framework
  - Review `TRAINING_README.md`
  - Implement custom trainer
  - Add custom metrics

## 📊 Monitoring Phase

- [ ] **Step 27**: Set up logging
  ```python
  from training_utils import setup_logging
  logger = setup_logging(log_dir='logs', log_level=logging.INFO)
  ```

- [ ] **Step 28**: Monitor GPU usage (if using CUDA)
  ```bash
  nvidia-smi -l 1  # Refresh every 1 second
  ```

- [ ] **Step 29**: Check checkpoint directory
  ```bash
  ls -lh checkpoints/*/best_model.pt
  ```

## 🎓 Learning Resources

- [ ] **Step 30**: Read module documentation
  - [ ] `config_management.py` header
  - [ ] `custom_trainer.py` header
  - [ ] `experiment_tracking.py` header
  - [ ] `hyperparameter_tuning.py` header

## ✨ Advanced Features to Explore

After completing basic setup:

- [ ] **Mixed Precision Training**
  ```yaml
  training:
    mixed_precision: true
  ```

- [ ] **Gradient Accumulation**
  ```yaml
  training:
    gradient_accumulation_steps: 4
  ```

- [ ] **Early Stopping**
  ```yaml
  training:
    early_stopping_patience: 10
  ```

- [ ] **Learning Rate Scheduling**
  ```yaml
  training:
    scheduler: cosine
    warmup_epochs: 5
  ```

- [ ] **Bayesian Hyperparameter Optimization**
  ```python
  from hyperparameter_tuning import HyperparameterTuner
  tuner = HyperparameterTuner()
  tuner.optimize(training_func, n_trials=100, n_jobs=4)
  ```

## 🐛 Troubleshooting Checklist

If you encounter issues:

- [ ] **CUDA not available?**
  - Check: `torch.cuda.is_available()`
  - Use CPU: `device: cpu` in config

- [ ] **Out of memory?**
  - Reduce batch_size in config
  - Enable gradient_accumulation_steps
  - Enable mixed_precision

- [ ] **Import errors?**
  - Reinstall: `pip install -r requirements_training.txt`
  - Check Python version: `python --version` (needs 3.9+)

- [ ] **Configuration validation error?**
  - Check YAML syntax
  - Review `config_template.yaml`
  - Validate with Python

- [ ] **Training very slow?**
  - Increase num_workers
  - Enable mixed_precision
  - Use GPU (cuda)

## 📁 File Organization

Before starting, organize your files:

```bash
d:\myfacedetect\
├── train\
│   ├── config_management.py
│   ├── custom_trainer.py
│   ├── experiment_tracking.py
│   ├── hyperparameter_tuning.py
│   ├── training_pipeline.py
│   ├── training_utils.py
│   ├── config_template.yaml
│   ├── requirements_training.txt
│   ├── README.md
│   └── ...
├── data\
│   └── your_dataset_here\
└── checkpoints\
    └── (auto-created during training)
```

## 📝 Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements_training.txt

# Run basic training
python training_pipeline.py --config config_template.yaml

# Run examples
python training_examples.py --example 0

# Run tests
pytest test_training_pipeline.py -v

# Check experiments
python -c "from experiment_tracking import ExperimentTracker; \
tracker = ExperimentTracker(); \
print(tracker.list_experiments())"

# Compare experiments
python -c "from experiment_tracking import ExperimentTracker; \
tracker = ExperimentTracker(); \
exps = [e['id'] for e in tracker.list_experiments()]; \
print(tracker.compare_experiments(exps))"

# Hyperparameter tuning
python training_examples.py --example 3
```

## ✅ Final Checklist

- [ ] All dependencies installed
- [ ] Examples run successfully
- [ ] Configuration created
- [ ] Basic training completed
- [ ] Results exported
- [ ] Tests passing
- [ ] Ready for advanced usage

## 🎉 Success Criteria

You've successfully set up the framework when:

✅ Installation completes without errors
✅ Examples run and produce output
✅ Training starts and loss decreases
✅ Results are saved in experiments/
✅ Experiment tracker lists your runs
✅ Tests pass (20+ test cases)
✅ Configuration loads successfully

## 📞 Getting Help

If stuck:

1. **Check Documentation**
   - README.md for quick help
   - TRAINING_README.md for details
   - REFERENCE.md for architecture

2. **Review Examples**
   - Run training_examples.py
   - Study example code
   - Try extensions

3. **Review Tests**
   - Look at test_training_pipeline.py
   - Find similar test case
   - Adapt for your use case

4. **Check Configuration**
   - Validate config_template.yaml
   - Review YAML syntax
   - Test with Python

## 🚀 Ready to Start?

```bash
cd d:\myfacedetect\train
pip install -r requirements_training.txt
python training_examples.py --example 0
```

**Estimated time to first successful training: 30 minutes**

Good luck! 🎉
