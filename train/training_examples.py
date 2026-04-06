"""Example usage of the training pipeline for face detection.

This script demonstrates:
1. Configuration management
2. Data preparation
3. Model training with custom trainer
4. Experiment tracking
5. Hyperparameter tuning
6. Results visualization
"""
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from config_management import ConfigManager, Config, DataConfig
from custom_trainer import CustomTrainer, TrainingConfig
from experiment_tracking import ExperimentTracker, CheckpointManager
from hyperparameter_tuning import HyperparameterTuner
from training_pipeline import TrainingPipeline


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceDetectionDataset(Dataset):
    """Example face detection dataset.
    
    In practice, this would load real face detection data.
    """
    
    def __init__(self, num_samples: int = 1000, img_size: int = 416):
        """Initialize dataset.
        
        Args:
            num_samples: Number of samples
            img_size: Image size
        """
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # Generate random image and label for demonstration
        # In practice, load actual image and annotations
        image = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, 2, (1,)).item()  # Binary: face/no-face
        return image, label


class SimpleFaceDetector(nn.Module):
    """Simple face detection model for demonstration.
    
    In practice, use YOLO, RetinaFace, or similar architectures.
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.1):
        """Initialize model.
        
        Args:
            num_classes: Number of classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Class logits (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def example_basic_training():
    """Example 1: Basic training with configuration."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Training")
    logger.info("=" * 60)
    
    # Create config
    config = Config(data=DataConfig(dataset_path='./dummy_data'))
    config.experiment_name = "face_detection_basic"
    config.training.num_epochs = 5  # Short example
    config.model.num_classes = 2
    
    # Prepare data
    dataset = FaceDetectionDataset(num_samples=200)
    train_size = 160
    val_size = 40
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    model = SimpleFaceDetector(num_classes=2, dropout=0.1)
    
    # Create trainer
    training_config = TrainingConfig(
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    trainer = CustomTrainer(model, training_config)
    
    # Train
    history = trainer.fit(train_loader, val_loader, save_best=True)
    
    logger.info(f"Training complete!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    return model, history


def example_experiment_tracking():
    """Example 2: Track multiple experiments."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Experiment Tracking")
    logger.info("=" * 60)
    
    tracker = ExperimentTracker()
    
    # Create experiments
    configs = [
        {'name': 'baseline', 'lr': 1e-3, 'batch_size': 32},
        {'name': 'high_lr', 'lr': 5e-3, 'batch_size': 32},
        {'name': 'large_batch', 'lr': 1e-3, 'batch_size': 64},
    ]
    
    for config in configs:
        exp_id = tracker.create_experiment(
            f"face_detection_{config['name']}",
            config,
            f"Experiment with {config['name']}"
        )
        
        tracker.start_experiment(exp_id)
        
        # Simulate training by logging metrics
        for epoch in range(5):
            # Simulate metric values
            loss = 2.0 - epoch * 0.2 + np.random.normal(0, 0.1)
            accuracy = 0.5 + epoch * 0.08 + np.random.normal(0, 0.05)
            
            tracker.log_metrics(
                exp_id,
                epoch,
                loss=loss,
                accuracy=accuracy
            )
        
        tracker.complete_experiment(exp_id, {
            'final_loss': loss,
            'final_accuracy': accuracy
        })
    
    # List experiments
    logger.info(f"\nAll experiments:")
    for exp in tracker.list_experiments():
        logger.info(f"  {exp['id']}: {exp['name']} - {exp['status']}")
    
    # Compare experiments
    exp_ids = [e['id'] for e in tracker.list_experiments()]
    comparison = tracker.compare_experiments(exp_ids)
    logger.info(f"\nComparison:\n{comparison}")
    
    # Export results
    tracker.export_results('face_detection_results.csv')
    logger.info("Results exported to face_detection_results.csv")


def example_hyperparameter_tuning():
    """Example 3: Hyperparameter tuning."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Hyperparameter Tuning")
    logger.info("=" * 60)
    
    # Create tuner
    tuner = HyperparameterTuner(study_name='face_detection_tuning')
    
    # Define training function
    def training_func(params, trial):
        """Training function returns metric."""
        # Simulate training with given parameters
        lr = params['lr']
        batch_size = params['batch_size']
        
        # Simulate performance based on hyperparameters
        # In practice, train actual model here
        base_mAP = 0.85
        
        # Learning rate effect
        lr_factor = 1.0 if 1e-4 <= lr <= 1e-2 else 0.8
        
        # Batch size effect
        bs_factor = 1.0 if batch_size >= 32 else 0.9
        
        final_mAP = base_mAP * lr_factor * bs_factor
        final_mAP += np.random.normal(0, 0.01)  # Add noise
        
        return max(0.0, min(1.0, final_mAP))  # Clamp to [0, 1]
    
    # Run optimization (with fewer trials for example)
    logger.info("Starting hyperparameter optimization...")
    tuner.optimize(training_func, n_trials=10, n_jobs=1)
    
    # Get results
    best_params = tuner.get_best_params()
    logger.info(f"\nBest parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save results
    results = tuner.save_results('tuning_results.json')
    logger.info(f"\nOptimization complete!")
    logger.info(f"Best trial: {results['best_trial']['number']}")
    logger.info(f"Best mAP: {results['best_trial']['value']:.4f}")


def example_metrics_computation():
    """Example 4: Custom metrics computation."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Custom Metrics")
    logger.info("=" * 60)
    
    def compute_face_detection_metrics(y_true, y_pred):
        """Compute custom metrics for face detection."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    # Generate dummy predictions
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    
    # Compute metrics
    metrics = compute_face_detection_metrics(y_true, y_pred)
    
    logger.info(f"Computed metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics


def example_training_pipeline():
    """Example 5: Complete training pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Complete Training Pipeline")
    logger.info("=" * 60)
    
    # Create config
    config = Config(data=DataConfig(dataset_path='./dummy_data'))
    config.experiment_name = "face_detection_pipeline"
    config.training.num_epochs = 5
    config.data.batch_size = 32
    
    # Create pipeline
    pipeline = TrainingPipeline(config)
    
    # Run pipeline
    results = pipeline.run()
    
    logger.info(f"\nPipeline Results:")
    logger.info(f"  Test Loss: {results['test_metrics'].get('loss', 'N/A'):.4f}")
    
    return results


def main():
    """Run all examples."""
    parser = argparse.ArgumentParser(description='Training pipeline examples')
    parser.add_argument('--example', type=int, default=0,
                       help='Example to run (0=all, 1-5=specific)')
    
    args = parser.parse_args()
    
    try:
        if args.example in [0, 1]:
            example_basic_training()
        
        if args.example in [0, 2]:
            example_experiment_tracking()
        
        if args.example in [0, 3]:
            example_hyperparameter_tuning()
        
        if args.example in [0, 4]:
            example_metrics_computation()
        
        if args.example in [0, 5]:
            example_training_pipeline()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == '__main__':
    main()
