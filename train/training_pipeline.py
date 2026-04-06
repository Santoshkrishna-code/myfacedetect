"""Comprehensive training pipeline orchestrator.

Features:
- Integrates data loading, model training, and evaluation
- Manages experiments and checkpoints
- Hyperparameter tuning
- Result analysis and visualization
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config_management import ConfigManager, Config
from custom_trainer import CustomTrainer, TrainingConfig
from experiment_tracking import ExperimentTracker, CheckpointManager
from hyperparameter_tuning import HyperparameterTuner


class TrainingPipeline:
    """Complete training pipeline."""
    
    def __init__(self, config: Config):
        """Initialize pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Setup
        self.logger = self._setup_logging()
        self.device = torch.device(config.training.device)
        
        # Managers
        self.experiment_tracker = ExperimentTracker()
        self.checkpoint_manager = CheckpointManager(config.checkpoints_dir)
        
        # Experiment ID
        self.exp_id = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('TrainingPipeline')
        logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def setup_experiment(self) -> str:
        """Setup experiment tracking.
        
        Returns:
            Experiment ID
        """
        exp_id = self.experiment_tracker.create_experiment(
            name=self.config.experiment_name,
            config=self._config_to_dict(),
            description=f"Training started at {datetime.now().isoformat()}"
        )
        
        self.exp_id = exp_id
        self.experiment_tracker.start_experiment(exp_id)
        
        self.logger.info(f"Experiment created: {exp_id}")
        
        return exp_id
    
    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self.config)
    
    def prepare_data(self) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Prepare data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Preparing data loaders...")
        
        # This is a placeholder - implement actual data loading
        # with your dataset
        
        # Example: Create dummy dataloaders for demonstration
        from torch.utils.data import TensorDataset, random_split
        
        # Create dummy dataset
        dummy_images = torch.randn(1000, 3, self.config.data.img_size, 
                                   self.config.data.img_size)
        dummy_labels = torch.randint(0, self.config.model.num_classes, (1000,))
        dataset = TensorDataset(dummy_images, dummy_labels)
        
        # Split dataset
        train_size = int(len(dataset) * self.config.data.train_split)
        val_size = int(len(dataset) * self.config.data.val_split)
        test_size = len(dataset) - train_size - val_size
        
        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        test_loader = DataLoader(
            test_set,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.logger.info(f"Data prepared: train={len(train_set)}, "
                        f"val={len(val_set)}, test={len(test_set)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> nn.Module:
        """Create model.
        
        Returns:
            Model instance
        """
        self.logger.info(f"Creating model: {self.config.model.name}")
        
        # Placeholder - implement actual model creation
        # Example using a simple ResNet
        try:
            import torchvision.models as models
            
            if self.config.model.backbone == 'resnet50':
                model = models.resnet50(
                    pretrained=self.config.model.pretrained,
                    num_classes=self.config.model.num_classes
                )
            else:
                raise ValueError(f"Unknown backbone: {self.config.model.backbone}")
            
        except Exception as e:
            self.logger.warning(f"Could not create model: {e}")
            # Fallback to simple model for testing
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.config.model.num_classes)
            )
        
        self.logger.info(f"Model created with {self._count_parameters(model)} parameters")
        
        return model
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             model: nn.Module,
             metric_fn: Optional[Callable] = None) -> Dict:
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model: Model to train
            metric_fn: Optional metric computation function
            
        Returns:
            Training history
        """
        self.logger.info("Starting training...")
        
        # Create trainer
        training_config = TrainingConfig(
            num_epochs=self.config.training.num_epochs,
            batch_size=self.config.data.batch_size,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            optimizer_type=self.config.training.optimizer,
            lr_scheduler_type=self.config.training.scheduler,
            warmup_epochs=self.config.training.warmup_epochs,
            dropout=self.config.model.dropout,
            label_smoothing=self.config.training.label_smoothing,
            use_mixed_precision=self.config.training.mixed_precision,
            accumulation_steps=self.config.training.gradient_accumulation_steps,
            early_stopping_patience=10,
            device=self.config.training.device,
            checkpoint_dir=self.config.checkpoints_dir,
            save_every_n_epochs=self.config.training.save_every_n_epochs,
            log_every_n_batches=self.config.training.log_every_n_batches
        )
        
        trainer = CustomTrainer(model, training_config, device=self.device)
        
        # Train
        history = trainer.fit(
            train_loader,
            val_loader,
            metric_fn=metric_fn,
            save_best=True
        )
        
        self.logger.info("Training complete!")
        
        # Save checkpoint
        if self.exp_id:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self._config_to_dict(),
                'history': history
            }
            self.checkpoint_manager.save_checkpoint(
                self.exp_id,
                checkpoint,
                'model_final.pt'
            )
        
        return history
    
    @torch.no_grad()
    def evaluate(self, 
                 test_loader: DataLoader, 
                 model: nn.Module,
                 metric_fn: Optional[Callable] = None) -> Dict:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            model: Trained model
            metric_fn: Optional metric computation function
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating on test set...")
        
        model.eval()
        model.to(self.device)
        
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        metrics = {'loss': avg_loss}
        
        if metric_fn:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            custom_metrics = metric_fn(all_labels, all_preds)
            metrics.update(custom_metrics)
        
        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        
        return metrics
    
    def run(self, metric_fn: Optional[Callable] = None) -> Dict:
        """Run complete training pipeline.
        
        Args:
            metric_fn: Optional metric computation function
            
        Returns:
            Final results dictionary
        """
        try:
            # Setup
            self.setup_experiment()
            
            # Prepare data
            train_loader, val_loader, test_loader = self.prepare_data()
            
            # Create model
            model = self.create_model()
            
            # Train
            history = self.train(train_loader, val_loader, model, metric_fn)
            
            # Evaluate
            test_metrics = self.evaluate(test_loader, model, metric_fn)
            
            # Save results
            results = {
                'config': self._config_to_dict(),
                'history': history,
                'test_metrics': test_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Mark experiment as complete
            if self.exp_id:
                self.experiment_tracker.complete_experiment(
                    self.exp_id,
                    test_metrics
                )
            
            self.logger.info("Pipeline complete!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Override with command-line arguments
    if args.exp_name:
        config.experiment_name = args.exp_name
    if args.device:
        config.training.device = args.device
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Test Loss: {results['test_metrics'].get('loss', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
