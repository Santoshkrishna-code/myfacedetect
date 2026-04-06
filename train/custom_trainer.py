"""Custom trainer class with advanced training features.

Features:
- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Checkpoint management
- Metrics tracking
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Tuple
import json
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import sys


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Optimization
    optimizer_type: str = 'adam'  # 'adam', 'sgd', 'adamw'
    momentum: float = 0.9
    
    # Learning rate scheduling
    lr_scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'step'
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 30
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Gradient accumulation
    accumulation_steps: int = 1
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 5
    
    # Logging
    log_every_n_batches: int = 100


class MetricsTracker:
    """Track training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
    
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            self.metrics[key].append(float(value))
    
    def epoch_reset(self):
        """Reset batch metrics at epoch end."""
        self.metrics.clear()
    
    def add_epoch(self, **kwargs):
        """Add epoch-level metrics."""
        for key, value in kwargs.items():
            self.epoch_metrics[key].append(float(value))
    
    def get_avg(self, key: str) -> float:
        """Get average of metric in current epoch."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.mean(self.metrics[key])
    
    def get_epoch_avg(self, key: str) -> float:
        """Get average across all epochs."""
        if key not in self.epoch_metrics or len(self.epoch_metrics[key]) == 0:
            return 0.0
        return np.mean(self.epoch_metrics[key])
    
    def get_all_epoch_metrics(self) -> Dict:
        """Get all epoch metrics."""
        return dict(self.epoch_metrics)


class CustomTrainer:
    """Custom training loop with advanced features."""
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 device: Optional[str] = None):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on (cpu/cuda)
        """
        self.model = model
        self.config = config
        self.device = device or config.device
        self.model.to(self.device)
        
        # Setup
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        
        # Mixed precision
        self.use_mixed_precision = config.use_mixed_precision and self.device == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Early stopping
        self.early_stopping_counter = 0
        self.best_metric = float('-inf')
        self.best_epoch = 0
        
        # Checkpointing
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        # Logging
        self.logger = self._setup_logging()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        params = self.model.parameters()
        
        if self.config.optimizer_type == 'adam':
            return optim.Adam(params, 
                            lr=self.config.learning_rate,
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == 'adamw':
            return optim.AdamW(params,
                             lr=self.config.learning_rate,
                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == 'sgd':
            return optim.SGD(params,
                           lr=self.config.learning_rate,
                           momentum=self.config.momentum,
                           weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs - self.config.warmup_epochs
            )
        elif self.config.lr_scheduler_type == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=self.config.num_epochs
            )
        elif self.config.lr_scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_steps,
                gamma=self.config.lr_decay_rate
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler_type}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        self.metrics.epoch_reset()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_every_n_batches == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch [{epoch}/{self.config.num_epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}, LR: {lr:.2e}"
                )
        
        # Update metrics
        avg_loss = total_loss / num_batches
        self.metrics.add_epoch(loss=avg_loss)
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, 
                epoch: int, 
                metric_fn: Optional[Callable] = None) -> Dict:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            metric_fn: Optional function to compute additional metrics
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            preds = outputs.argmax(dim=1)
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        metrics = {'loss': avg_loss}
        
        if metric_fn is not None:
            all_preds = np.concatenate(all_predictions)
            all_labels_arr = np.concatenate(all_labels)
            custom_metrics = metric_fn(all_labels_arr, all_preds)
            metrics.update(custom_metrics)
        
        self.logger.info(
            f"Epoch [{epoch}/{self.config.num_epochs}] "
            f"Val Loss: {avg_loss:.4f}"
        )
        
        return metrics
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            metric_fn: Optional[Callable] = None,
            save_best: bool = True) -> Dict:
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            metric_fn: Optional function to compute validation metrics
            save_best: Whether to save best model
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate(val_loader, epoch, metric_fn)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling (after warmup)
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Early stopping check
            current_metric = val_metrics.get('loss', -val_metrics.get('accuracy', 0))
            if current_metric > self.best_metric + self.config.early_stopping_min_delta:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                
                if save_best:
                    self.save_checkpoint(f'best_model.pt', is_best=True)
            else:
                self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch}. "
                        f"Best epoch: {self.best_epoch}"
                    )
                    break
            
            # Periodic checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        self.logger.info(f"Training complete! Best epoch: {self.best_epoch}")
        return history
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'is_best': is_best
        }
        
        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.info(f"Checkpoint loaded: {path}")
