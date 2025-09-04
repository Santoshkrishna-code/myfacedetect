"""
Centralized logging configuration for MyFaceDetect v0.3.0
Provides clean, structured logging with reduced noise from third-party libraries.
"""
import logging
import sys
import os
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    quiet_third_party: bool = True
) -> logging.Logger:
    """
    Configure comprehensive logging for MyFaceDetect.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
        log_file: Optional file to write logs to
        quiet_third_party: Reduce noise from third-party libraries

    Returns:
        Configured logger instance
    """

    # Set up root logger level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create custom formatter
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)8s] %(name)s: %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%H:%M:%S')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file {log_file}: {e}")

    # Quiet third-party libraries to reduce noise
    if quiet_third_party:
        quiet_loggers = [
            'ultralytics',
            'torch',
            'torchvision',
            'PIL',
            'matplotlib',
            'insightface',
            'onnxruntime',
            'cv2',
            'numpy',
            'urllib3',
            'requests'
        ]

        for logger_name in quiet_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Get MyFaceDetect logger
    logger = logging.getLogger('myfacedetect')
    logger.setLevel(numeric_level)

    return logger


def setup_cpu_only_logging():
    """Configure logging specifically for CPU-only execution with reduced noise."""

    # Suppress CUDA-related warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow

    # Set up clean logging
    logger = setup_logging(
        level="INFO",
        format_string='%(asctime)s [%(levelname)7s] %(name)20s: %(message)s',
        quiet_third_party=True
    )

    # Additional third-party noise reduction
    extra_quiet = [
        'tensorflow',
        'absl',
        'h5py',
        'keras',
        'sklearn',
        'scipy'
    ]

    for logger_name in extra_quiet:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Specific ONNX and CUDA suppression
    logging.getLogger('onnxruntime').setLevel(logging.ERROR)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent naming convention."""
    return logging.getLogger(f'myfacedetect.{name}')


# Pre-configured loggers for different components
def get_detection_logger() -> logging.Logger:
    """Get logger for detection modules."""
    return get_logger('detection')


def get_recognition_logger() -> logging.Logger:
    """Get logger for recognition modules."""
    return get_logger('recognition')


def get_utils_logger() -> logging.Logger:
    """Get logger for utility modules."""
    return get_logger('utils')


def get_test_logger() -> logging.Logger:
    """Get logger for test modules."""
    return get_logger('test')
