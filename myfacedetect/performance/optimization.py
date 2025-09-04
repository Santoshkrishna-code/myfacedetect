"""
GPU Acceleration Module
GPU acceleration support for face detection and recognition.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """GPU acceleration for face processing operations."""

    def __init__(self):
        self.cuda_available = False
        self.opencl_available = False
        self.device_info = {}

        self._detect_gpu_support()

    def _detect_gpu_support(self):
        """Detect available GPU acceleration options."""
        try:
            # Check CUDA support
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.cuda_available = True
                self.device_info['cuda_devices'] = cv2.cuda.getCudaEnabledDeviceCount(
                )
                logger.info(
                    f"CUDA support detected: {self.device_info['cuda_devices']} devices")

            # Check OpenCL support
            if cv2.ocl.haveOpenCL():
                self.opencl_available = True
                self.device_info['opencl_available'] = True
                cv2.ocl.setUseOpenCL(True)
                logger.info("OpenCL support detected")

            if not self.cuda_available and not self.opencl_available:
                logger.info("No GPU acceleration available, using CPU")

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

    def upload_to_gpu(self, image: np.ndarray) -> Any:
        """Upload image to GPU memory."""
        try:
            if self.cuda_available:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                return gpu_image
            elif self.opencl_available:
                return cv2.UMat(image)
            else:
                return image
        except Exception as e:
            logger.warning(f"GPU upload failed: {e}")
            return image

    def download_from_gpu(self, gpu_image: Any) -> np.ndarray:
        """Download image from GPU memory."""
        try:
            if self.cuda_available and isinstance(gpu_image, cv2.cuda_GpuMat):
                result = gpu_image.download()
                return result
            elif self.opencl_available and isinstance(gpu_image, cv2.UMat):
                return gpu_image.get()
            else:
                return gpu_image
        except Exception as e:
            logger.warning(f"GPU download failed: {e}")
            return gpu_image

    def gpu_resize(self, image: Any, size: Tuple[int, int]) -> Any:
        """Resize image on GPU."""
        try:
            if self.cuda_available and isinstance(image, cv2.cuda_GpuMat):
                gpu_resized = cv2.cuda_GpuMat()
                cv2.cuda.resize(image, size, gpu_resized)
                return gpu_resized
            elif self.opencl_available:
                if isinstance(image, cv2.UMat):
                    return cv2.resize(image, size)
                else:
                    umat_image = cv2.UMat(image)
                    return cv2.resize(umat_image, size)
            else:
                return cv2.resize(image, size)
        except Exception as e:
            logger.warning(f"GPU resize failed: {e}")
            return cv2.resize(self.download_from_gpu(image), size)

    def gpu_blur(self, image: Any, kernel_size: Tuple[int, int]) -> Any:
        """Apply Gaussian blur on GPU."""
        try:
            if self.cuda_available and isinstance(image, cv2.cuda_GpuMat):
                gpu_blurred = cv2.cuda_GpuMat()
                cv2.cuda.GaussianBlur(image, kernel_size, 0, gpu_blurred)
                return gpu_blurred
            elif self.opencl_available:
                if isinstance(image, cv2.UMat):
                    return cv2.GaussianBlur(image, kernel_size, 0)
                else:
                    umat_image = cv2.UMat(image)
                    return cv2.GaussianBlur(umat_image, kernel_size, 0)
            else:
                return cv2.GaussianBlur(image, kernel_size, 0)
        except Exception as e:
            logger.warning(f"GPU blur failed: {e}")
            return cv2.GaussianBlur(self.download_from_gpu(image), kernel_size, 0)

    def gpu_convert_color(self, image: Any, conversion_code: int) -> Any:
        """Convert color space on GPU."""
        try:
            if self.cuda_available and isinstance(image, cv2.cuda_GpuMat):
                gpu_converted = cv2.cuda_GpuMat()
                cv2.cuda.cvtColor(image, conversion_code, gpu_converted)
                return gpu_converted
            elif self.opencl_available:
                if isinstance(image, cv2.UMat):
                    return cv2.cvtColor(image, conversion_code)
                else:
                    umat_image = cv2.UMat(image)
                    return cv2.cvtColor(umat_image, conversion_code)
            else:
                return cv2.cvtColor(image, conversion_code)
        except Exception as e:
            logger.warning(f"GPU color conversion failed: {e}")
            return cv2.cvtColor(self.download_from_gpu(image), conversion_code)

    def create_gpu_cascade(self, cascade_file: str) -> Optional[Any]:
        """Create GPU-accelerated cascade classifier."""
        try:
            if self.cuda_available:
                return cv2.cuda.CascadeClassifier_create(cascade_file)
            else:
                return cv2.CascadeClassifier(cascade_file)
        except Exception as e:
            logger.warning(f"GPU cascade creation failed: {e}")
            return cv2.CascadeClassifier(cascade_file)

    def gpu_detect_faces(self, cascade: Any, image: Any, scale_factor: float = 1.1,
                         min_neighbors: int = 3) -> List[Tuple[int, int, int, int]]:
        """Detect faces using GPU acceleration."""
        try:
            if self.cuda_available and hasattr(cascade, 'detectMultiScale'):
                # CUDA cascade detection
                faces_gpu = cascade.detectMultiScale(image)
                faces = faces_gpu.download() if hasattr(faces_gpu, 'download') else faces_gpu
                return [tuple(face) for face in faces] if len(faces) > 0 else []
            else:
                # CPU cascade detection
                cpu_image = self.download_from_gpu(image)
                faces = cascade.detectMultiScale(
                    cpu_image, scale_factor, min_neighbors)
                return [tuple(face) for face in faces]
        except Exception as e:
            logger.warning(f"GPU face detection failed: {e}")
            cpu_image = self.download_from_gpu(image)
            faces = cascade.detectMultiScale(
                cpu_image, scale_factor, min_neighbors)
            return [tuple(face) for face in faces]

    def benchmark_gpu_performance(self, test_image: np.ndarray, iterations: int = 100) -> Dict[str, float]:
        """Benchmark GPU vs CPU performance."""
        results = {}

        # CPU benchmark
        start_time = time.time()
        for _ in range(iterations):
            resized = cv2.resize(test_image, (224, 224))
            blurred = cv2.GaussianBlur(resized, (15, 15), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        cpu_time = time.time() - start_time
        results['cpu_time'] = cpu_time

        # GPU benchmark (if available)
        if self.cuda_available or self.opencl_available:
            gpu_image = self.upload_to_gpu(test_image)

            start_time = time.time()
            for _ in range(iterations):
                resized = self.gpu_resize(gpu_image, (224, 224))
                blurred = self.gpu_blur(resized, (15, 15))
                gray = self.gpu_convert_color(blurred, cv2.COLOR_BGR2GRAY)
                final = self.download_from_gpu(gray)
            gpu_time = time.time() - start_time
            results['gpu_time'] = gpu_time
            results['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 1.0

        return results

    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        info = self.device_info.copy()
        info['cuda_available'] = self.cuda_available
        info['opencl_available'] = self.opencl_available

        try:
            if self.cuda_available:
                info['cuda_version'] = cv2.cuda.getCudaEnabledDeviceCount()

            if self.opencl_available:
                # Get OpenCL device info (simplified)
                info['opencl_platforms'] = "Available"
        except Exception as e:
            logger.warning(f"Device info collection failed: {e}")

        return info


class ModelOptimizer:
    """Model optimization for faster inference."""

    def __init__(self):
        self.onnx_available = False
        self.tensorrt_available = False
        self._check_optimization_support()

    def _check_optimization_support(self):
        """Check available optimization frameworks."""
        try:
            import onnxruntime
            self.onnx_available = True
            logger.info("ONNX Runtime available for model optimization")
        except ImportError:
            pass

        try:
            import tensorrt
            self.tensorrt_available = True
            logger.info("TensorRT available for model optimization")
        except ImportError:
            pass

    def optimize_model_for_inference(self, model_path: str, optimization_level: str = 'medium') -> Optional[str]:
        """
        Optimize model for faster inference.

        Args:
            model_path: Path to model file
            optimization_level: 'low', 'medium', 'high'

        Returns:
            Path to optimized model or None if failed
        """
        try:
            if not self.onnx_available:
                logger.warning("ONNX Runtime not available for optimization")
                return None

            import onnxruntime as ort

            # Set optimization level
            if optimization_level == 'low':
                graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            elif optimization_level == 'medium':
                graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            else:  # high
                graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = graph_optimization_level
            sess_options.optimized_model_filepath = model_path.replace(
                '.onnx', '_optimized.onnx')

            # Create session to trigger optimization
            session = ort.InferenceSession(model_path, sess_options)

            logger.info(
                f"Model optimized: {sess_options.optimized_model_filepath}")
            return sess_options.optimized_model_filepath

        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return None

    def quantize_model(self, model_path: str, quantization_mode: str = 'dynamic') -> Optional[str]:
        """
        Quantize model for faster inference and smaller size.

        Args:
            model_path: Path to ONNX model
            quantization_mode: 'dynamic' or 'static'

        Returns:
            Path to quantized model or None if failed
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_model_path = model_path.replace(
                '.onnx', '_quantized.onnx')

            if quantization_mode == 'dynamic':
                quantize_dynamic(
                    model_input=model_path,
                    model_output=quantized_model_path,
                    weight_type=QuantType.QInt8
                )
            else:
                logger.warning("Static quantization not implemented")
                return None

            logger.info(f"Model quantized: {quantized_model_path}")
            return quantized_model_path

        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return None


class PerformanceProfiler:
    """Performance profiling and monitoring."""

    def __init__(self):
        self.profiles = {}

    def profile_function(self, func_name: str):
        """Decorator for profiling function performance."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()

                execution_time = end_time - start_time

                if func_name not in self.profiles:
                    self.profiles[func_name] = []
                self.profiles[func_name].append(execution_time)

                return result
            return wrapper
        return decorator

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for profiled functions."""
        stats = {}

        for func_name, times in self.profiles.items():
            if times:
                stats[func_name] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_calls': len(times),
                    'total_time': np.sum(times)
                }

        return stats

    def reset_profiles(self):
        """Reset all performance profiles."""
        self.profiles.clear()

    def benchmark_detection_pipeline(self, detector, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """Benchmark complete detection pipeline."""
        results = {
            'total_images': len(test_images),
            'total_faces_detected': 0,
            'processing_times': [],
            'fps_estimates': []
        }

        for image in test_images:
            start_time = time.time()

            # Run detection
            faces = detector.detect_faces(image)

            end_time = time.time()
            processing_time = end_time - start_time

            results['processing_times'].append(processing_time)
            results['total_faces_detected'] += len(faces)

            if processing_time > 0:
                fps = 1.0 / processing_time
                results['fps_estimates'].append(fps)

        # Calculate statistics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(
                results['processing_times'])
            results['min_processing_time'] = np.min(
                results['processing_times'])
            results['max_processing_time'] = np.max(
                results['processing_times'])

        if results['fps_estimates']:
            results['avg_fps'] = np.mean(results['fps_estimates'])
            results['min_fps'] = np.min(results['fps_estimates'])
            results['max_fps'] = np.max(results['fps_estimates'])

        return results


def create_gpu_accelerator() -> GPUAccelerator:
    """Create a GPU accelerator instance."""
    return GPUAccelerator()


def create_model_optimizer() -> ModelOptimizer:
    """Create a model optimizer instance."""
    return ModelOptimizer()


def create_performance_profiler() -> PerformanceProfiler:
    """Create a performance profiler instance."""
    return PerformanceProfiler()
