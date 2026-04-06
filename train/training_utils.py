"""Utility functions for training pipeline.

Features:
- Checksum/hashing utilities
- Data normalization
- Visualization helpers
- Statistics computation
- Device utilities
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path
import hashlib
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# Device Utilities
# ============================================================================

def get_device(device: Optional[str] = None, verbose: bool = True) -> torch.device:
    """Get torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', None for auto)
        verbose: Print device info
        
    Returns:
        Torch device object
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
    
    device_obj = torch.device(device)
    
    if verbose:
        logger.info(f"Using device: {device_obj}")
        if device_obj.type == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device_obj


def get_device_stats() -> Dict:
    """Get current device statistics.
    
    Returns:
        Dictionary of device stats
    """
    stats = {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        stats['cuda_version'] = torch.version.cuda
        stats['cudnn_version'] = torch.backends.cudnn.version()
        stats['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        stats['allocated_memory_gb'] = torch.cuda.memory_allocated() / 1e9
        stats['cached_memory_gb'] = torch.cuda.memory_reserved() / 1e9
    
    return stats


def clear_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


# ============================================================================
# Data Utilities
# ============================================================================

def compute_normalization_stats(loader: torch.utils.data.DataLoader) -> Tuple[List, List]:
    """Compute mean and std for dataset normalization.
    
    Args:
        loader: Data loader
        
    Returns:
        Tuple of (mean, std) lists
    """
    logger.info("Computing normalization statistics...")
    
    images = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images.append(batch[0])
        else:
            images.append(batch)
    
    images = torch.cat(images, dim=0)
    
    # Compute per-channel statistics
    # Assuming images are (N, C, H, W)
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    
    mean = mean.tolist()
    std = std.tolist()
    
    logger.info(f"Mean: {[f'{m:.4f}' for m in mean]}")
    logger.info(f"Std: {[f'{s:.4f}' for s in std]}")
    
    return mean, std


def normalize_image(image: torch.Tensor, 
                   mean: List[float],
                   std: List[float]) -> torch.Tensor:
    """Normalize image with mean and std.
    
    Args:
        image: Image tensor (C, H, W) or (B, C, H, W)
        mean: Mean values per channel
        std: Std values per channel
        
    Returns:
        Normalized image
    """
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    return (image - mean_tensor) / std_tensor


def denormalize_image(image: torch.Tensor,
                     mean: List[float],
                     std: List[float]) -> torch.Tensor:
    """Denormalize image.
    
    Args:
        image: Normalized image tensor
        mean: Mean values per channel
        std: Std values per channel
        
    Returns:
        Denormalized image
    """
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    return image * std_tensor + mean_tensor


# ============================================================================
# Statistics and Metrics
# ============================================================================

def compute_statistics(values: List[float]) -> Dict:
    """Compute statistics for a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary of statistics
    """
    values_arr = np.array(values)
    
    return {
        'mean': float(np.mean(values_arr)),
        'std': float(np.std(values_arr)),
        'min': float(np.min(values_arr)),
        'max': float(np.max(values_arr)),
        'median': float(np.median(values_arr)),
        'q25': float(np.percentile(values_arr, 25)),
        'q75': float(np.percentile(values_arr, 75))
    }


def compute_moving_average(values: List[float], window: int = 10) -> List[float]:
    """Compute moving average.
    
    Args:
        values: List of values
        window: Window size
        
    Returns:
        List of moving averages
    """
    if len(values) < window:
        return values
    
    ma = np.convolve(values, np.ones(window)/window, mode='valid').tolist()
    return [values[0]] * (window - 1) + ma


# ============================================================================
# File Utilities
# ============================================================================

def compute_file_hash(filepath: str, algorithm: str = 'md5') -> str:
    """Compute hash of file.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('md5', 'sha256')
        
    Returns:
        Hash string
    """
    if algorithm == 'md5':
        hash_func = hashlib.md5()
    elif algorithm == 'sha256':
        hash_func = hashlib.sha256()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def compute_dict_hash(data: Dict, algorithm: str = 'md5') -> str:
    """Compute hash of dictionary.
    
    Args:
        data: Dictionary
        algorithm: Hash algorithm
        
    Returns:
        Hash string
    """
    json_str = json.dumps(data, sort_keys=True)
    
    if algorithm == 'md5':
        return hashlib.md5(json_str.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(json_str.encode()).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def get_checkpoint_size(checkpoint_path: str) -> float:
    """Get checkpoint file size in MB.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        File size in MB
    """
    size_bytes = Path(checkpoint_path).stat().st_size
    return size_bytes / (1024 * 1024)


def estimate_model_size(model: torch.nn.Module) -> float:
    """Estimate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Estimated size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming float32 (4 bytes per parameter)
    size_bytes = total_params * 4
    return size_bytes / (1024 * 1024)


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Metrics plot
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        metrics = history['val_metrics']
        
        # Extract metrics
        metric_names = {}
        for epoch_metrics in metrics:
            for key, value in epoch_metrics.items():
                if key != 'loss':
                    if key not in metric_names:
                        metric_names[key] = []
                    metric_names[key].append(value)
        
        for metric_name, values in metric_names.items():
            axes[1].plot(values, label=metric_name, marker='o')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_metric_distribution(metrics: List[float], 
                            title: str = 'Metric Distribution',
                            save_path: Optional[str] = None):
    """Plot histogram of metric values.
    
    Args:
        metrics: List of metric values
        title: Plot title
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    plt.figure(figsize=(10, 5))
    plt.hist(metrics, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Plot saved to {save_path}")
    
    return plt.gcf()


# ============================================================================
# Logging Utilities
# ============================================================================

def setup_logging(log_dir: str = 'logs', 
                 log_level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    import sys
    from datetime import datetime
    
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('TrainingPipeline')
    logger.setLevel(log_level)
    
    # File handler
    log_file = Path(log_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to {log_file}")
    
    return logger


if __name__ == '__main__':
    # Example usage
    device = get_device()
    stats = get_device_stats()
    print(f"Device stats: {stats}")
