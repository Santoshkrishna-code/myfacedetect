"""
ONNX Runtime utilities for MyFaceDetect.
Provides CPU-only inference sessions and handles provider setup.
"""
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_safe_providers(force_cpu: bool = False) -> List[str]:
    """
    Get safe ONNX Runtime providers.

    Args:
        force_cpu: If True, only return CPU provider

    Returns:
        List of provider names in priority order
    """
    providers = ['CPUExecutionProvider']

    if not force_cpu:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()

            # Add GPU providers if available and not forcing CPU
            if 'CUDAExecutionProvider' in available:
                providers.insert(0, 'CUDAExecutionProvider')
            if 'TensorrtExecutionProvider' in available:
                providers.insert(0, 'TensorrtExecutionProvider')

        except Exception as e:
            logger.debug(f"Could not check ONNX providers: {e}")

    return providers


def create_inference_session(model_path: str, force_cpu: bool = None):
    """
    Create ONNX Runtime inference session with safe provider configuration.

    Args:
        model_path: Path to ONNX model file
        force_cpu: Force CPU-only execution (defaults to environment variable)

    Returns:
        onnxruntime.InferenceSession
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required. Install with: pip install onnxruntime")

    # Check if we should force CPU mode
    if force_cpu is None:
        force_cpu = os.environ.get(
            'MYFACEDETECT_CPU_ONLY', 'false').lower() == 'true'

    providers = get_safe_providers(force_cpu=force_cpu)

    try:
        session = ort.InferenceSession(model_path, providers=providers)
        logger.debug(
            f"Created ONNX session with providers: {session.get_providers()}")
        return session
    except Exception as e:
        # Fallback to CPU only
        logger.warning(
            f"Failed to create session with providers {providers}, falling back to CPU")
        session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        return session


def setup_cpu_only_mode():
    """Setup environment for CPU-only ONNX execution."""
    os.environ['MYFACEDETECT_CPU_ONLY'] = 'true'
    os.environ['OMP_NUM_THREADS'] = '1'

    # Suppress ONNX Runtime CUDA warnings
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)

    logger.info("Configured ONNX Runtime for CPU-only execution")


def check_gpu_availability() -> dict:
    """
    Check GPU availability for ONNX Runtime.

    Returns:
        Dictionary with GPU status information
    """
    status = {
        'cuda_available': False,
        'tensorrt_available': False,
        'providers': [],
        'recommended_setup': 'CPU-only'
    }

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        status['providers'] = providers

        if 'CUDAExecutionProvider' in providers:
            status['cuda_available'] = True
            status['recommended_setup'] = 'GPU (CUDA)'

        if 'TensorrtExecutionProvider' in providers:
            status['tensorrt_available'] = True
            status['recommended_setup'] = 'GPU (TensorRT)'

    except ImportError:
        status['providers'] = ['onnxruntime not installed']

    return status
