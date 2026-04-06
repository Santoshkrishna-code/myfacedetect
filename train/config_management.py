"""Configuration management for training pipeline.

Features:
- YAML/JSON configuration loading
- Configuration validation
- Environment variable interpolation
- Configuration merging
- Schema validation
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
from dataclasses import dataclass, field, asdict
import logging
from dotenv import load_dotenv
import jsonschema
from jsonschema import validate, ValidationError


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    augmentation: bool = True
    
    # Image settings
    img_size: int = 416
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = 'yolov8'
    pretrained: bool = True
    num_classes: int = 80
    input_channels: int = 3
    dropout: float = 0.1
    
    # Architecture
    backbone: str = 'resnet50'
    num_layers: int = 50
    
    # Weights
    weights_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Optimizer
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    
    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'linear', 'step'
    warmup_epochs: int = 5
    
    # Loss
    loss_fn: str = 'cross_entropy'
    label_smoothing: float = 0.0
    
    # Device
    device: str = 'cuda'
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n: int = 3
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Logging
    log_every_n_batches: int = 100
    log_dir: str = 'logs'


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotation: float = 10.0  # degrees
    scale: float = 0.2
    shear: float = 10.0  # degrees
    brightness: float = 0.2
    contrast: float = 0.2
    hue: float = 0.1
    saturation: float = 0.2
    blur: float = 0.1


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Experiment tracking
    experiment_name: str = 'default'
    seed: int = 42
    log_to_wandb: bool = False
    
    # Paths
    config_path: str = ''
    checkpoints_dir: str = 'checkpoints'
    output_dir: str = 'outputs'


class ConfigManager:
    """Manage training configurations."""
    
    # JSON schema for configuration validation
    CONFIG_SCHEMA = {
        'type': 'object',
        'properties': {
            'data': {
                'type': 'object',
                'properties': {
                    'dataset_path': {'type': 'string'},
                    'batch_size': {'type': 'integer', 'minimum': 1},
                    'num_workers': {'type': 'integer', 'minimum': 0},
                    'img_size': {'type': 'integer', 'minimum': 32},
                },
                'required': ['dataset_path']
            },
            'model': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'num_classes': {'type': 'integer', 'minimum': 1},
                },
                'required': ['name']
            },
            'training': {
                'type': 'object',
                'properties': {
                    'num_epochs': {'type': 'integer', 'minimum': 1},
                    'learning_rate': {'type': 'number', 'minimum': 0},
                    'batch_size': {'type': 'integer', 'minimum': 1},
                }
            }
        },
        'required': ['data', 'model', 'training']
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize config manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        load_dotenv()  # Load environment variables
    
    def load_config(self, config_path: str) -> Config:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Config object
            
        Raises:
            FileNotFoundError: If config file not found
            ValidationError: If config is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load file
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                config_dict = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Interpolate environment variables
        config_dict = self._interpolate_env_vars(config_dict)
        
        # Validate
        self._validate_config(config_dict)
        
        # Create Config object
        config = self._dict_to_config(config_dict)
        config.config_path = str(config_path)
        
        self.logger.info(f"Config loaded from {config_path}")
        
        return config
    
    def _interpolate_env_vars(self, obj: Any) -> Any:
        """Replace env var references with actual values.
        
        Args:
            obj: Object to interpolate (dict, list, or str)
            
        Returns:
            Interpolated object
        """
        if isinstance(obj, str):
            # Replace ${VAR} with environment variable
            if obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return os.getenv(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: self._interpolate_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_env_vars(item) for item in obj]
        return obj
    
    def _validate_config(self, config_dict: Dict):
        """Validate configuration against schema.
        
        Args:
            config_dict: Configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validate(instance=config_dict, schema=self.CONFIG_SCHEMA)
            self.logger.info("Configuration validation passed")
        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e.message}")
            raise
    
    def _dict_to_config(self, config_dict: Dict) -> Config:
        """Convert dictionary to Config object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
        """
        config = Config()
        
        # Update data config
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # Update model config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Update training config
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        # Update augmentation config
        if 'augmentation' in config_dict:
            for key, value in config_dict['augmentation'].items():
                if hasattr(config.augmentation, key):
                    setattr(config.augmentation, key, value)
        
        # Update top-level properties
        for key in ['experiment_name', 'seed', 'log_to_wandb', 
                   'checkpoints_dir', 'output_dir']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def save_config(self, config: Config, output_path: str, format: str = 'yaml'):
        """Save configuration to file.
        
        Args:
            config: Config object
            output_path: Output file path
            format: 'json' or 'yaml'
        """
        config_dict = asdict(config)
        
        with open(output_path, 'w') as f:
            if format == 'json':
                json.dump(config_dict, f, indent=2)
            elif format == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Config saved to {output_path}")
    
    def merge_configs(self, base_config: Config, override_dict: Dict) -> Config:
        """Merge override configuration into base configuration.
        
        Args:
            base_config: Base configuration
            override_dict: Dictionary of values to override
            
        Returns:
            Merged config
        """
        config_dict = asdict(base_config)
        
        # Merge nested dictionaries
        def merge_dict(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        config_dict = merge_dict(config_dict, override_dict)
        return self._dict_to_config(config_dict)


# Example configuration template
EXAMPLE_CONFIG_YAML = """
# Data configuration
data:
  dataset_path: ./data/coco
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  batch_size: 32
  num_workers: 4
  img_size: 416
  
# Model configuration
model:
  name: yolov8
  pretrained: true
  num_classes: 80
  backbone: resnet50
  
# Training configuration
training:
  num_epochs: 100
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine
  warmup_epochs: 5
  mixed_precision: true
  
# Augmentation configuration
augmentation:
  horizontal_flip: 0.5
  brightness: 0.2
  contrast: 0.2
  
# Experiment settings
experiment_name: yolov8_baseline
seed: 42
log_to_wandb: false
"""


if __name__ == '__main__':
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create example config
    config_path = 'example_config.yaml'
    with open(config_path, 'w') as f:
        f.write(EXAMPLE_CONFIG_YAML)
    
    # Load and display
    manager = ConfigManager(logger)
    config = manager.load_config(config_path)
    
    print("Configuration loaded:")
    print(f"  Dataset: {config.data.dataset_path}")
    print(f"  Model: {config.model.name}")
    print(f"  Epochs: {config.training.num_epochs}")
