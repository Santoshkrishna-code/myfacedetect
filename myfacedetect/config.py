"""
Configuration management for MyFaceDetect library.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for face detection parameters."""

    DEFAULT_CONFIG = {
        "haar_cascade": {
            "scale_factor": 1.1,
            "min_neighbors": 4,
            "min_size": [30, 30],
            "cascade_file": "haarcascade_frontalface_default.xml"
        },
        "mediapipe": {
            "model_selection": 0,
            "min_detection_confidence": 0.5
        },
        "realtime": {
            "fps_update_frames": 30,
            "default_window_name": "MyFaceDetect - Real-time",
            "mirror_camera": True,
            "show_fps": True,
            "show_controls": True
        },
        "detection": {
            "overlap_threshold": 0.3,
            "default_method": "mediapipe"
        },
        "output": {
            "default_save_dir": "detections",
            "face_crop_dir": "face_crops",
            "image_format": "jpg",
            "video_format": "mp4"
        }
    }

    def __init__(self, config_file: Optional[str] = None):
        self._config = self.DEFAULT_CONFIG.copy()
        self.config_file = config_file or "myfacedetect_config.json"
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file if it exists."""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                self._merge_config(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load config from {config_path}: {e}")

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")

    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """Merge user configuration with defaults."""
        for section, values in user_config.items():
            if section in self._config and isinstance(values, dict):
                self._config[section].update(values)
            else:
                self._config[section] = values

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = self.DEFAULT_CONFIG.copy()
        logger.info("Configuration reset to defaults")

    @property
    def haar_params(self) -> Dict[str, Any]:
        """Get Haar cascade parameters."""
        return self._config["haar_cascade"]

    @property
    def mediapipe_params(self) -> Dict[str, Any]:
        """Get MediaPipe parameters."""
        return self._config["mediapipe"]

    @property
    def realtime_params(self) -> Dict[str, Any]:
        """Get real-time detection parameters."""
        return self._config["realtime"]


# Global configuration instance
config = Config()
