"""
Configuration Manager for MyFaceDetect
Handles loading and validation of configuration files.
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.config = self._load_config()
        self.current_pipeline = "default"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(
                f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            "default": {
                "detector": "mediapipe",
                "recognizer": None,
                "preprocessing": ["alignment", "normalization"],
                "postprocessing": {
                    "nms_threshold": 0.4,
                    "confidence_threshold": 0.5
                },
                "performance": {
                    "device": "auto",
                    "batch_size": 1,
                    "precision": "fp32"
                }
            }
        }

    def get_pipeline_config(self, pipeline: str = None) -> Dict[str, Any]:
        """Get configuration for specified pipeline."""
        pipeline = pipeline or self.current_pipeline
        if pipeline not in self.config:
            logger.warning(f"Pipeline '{pipeline}' not found, using default")
            pipeline = "default"
        return self.config[pipeline]

    def set_pipeline(self, pipeline: str):
        """Set current pipeline."""
        if pipeline in self.config:
            self.current_pipeline = pipeline
            logger.info(f"Pipeline set to: {pipeline}")
        else:
            logger.error(f"Pipeline '{pipeline}' not found")
            raise ValueError(f"Pipeline '{pipeline}' not found")

    def list_pipelines(self) -> list:
        """List available pipelines."""
        return list(self.config.keys())

    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def update_config(self, pipeline: str, key: str, value: Any):
        """Update configuration value."""
        if pipeline not in self.config:
            self.config[pipeline] = {}

        keys = key.split('.')
        config_section = self.config[pipeline]
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
        config_section[keys[-1]] = value

        logger.info(f"Updated {pipeline}.{key} = {value}")


# Global config instance
config = ConfigManager()
