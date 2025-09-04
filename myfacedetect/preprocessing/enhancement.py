"""
Image Enhancement Module
Advanced preprocessing techniques for improving face detection and recognition.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Advanced image enhancement for face processing."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gaussian_blur_size = (5, 5)
        self.bilateral_filter_params = (9, 75, 75)

    def enhance_lighting(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance lighting conditions.

        Args:
            image: Input image
            method: Enhancement method ('clahe', 'gamma', 'histogram_eq', 'adaptive')

        Returns:
            Enhanced image
        """
        try:
            if method == 'clahe':
                return self._apply_clahe(image)
            elif method == 'gamma':
                return self._gamma_correction(image, gamma=1.2)
            elif method == 'histogram_eq':
                return self._histogram_equalization(image)
            elif method == 'adaptive':
                return self._adaptive_enhancement(image)
            else:
                logger.warning(f"Unknown enhancement method: {method}")
                return image
        except Exception as e:
            logger.warning(f"Lighting enhancement failed: {e}")
            return image

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)

    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction."""
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                               for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, gamma_table)

    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)

    def _adaptive_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive enhancement based on image statistics."""
        # Calculate image brightness
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        mean_brightness = np.mean(gray)

        if mean_brightness < 100:  # Dark image
            return self._gamma_correction(image, gamma=0.7)
        elif mean_brightness > 180:  # Bright image
            return self._gamma_correction(image, gamma=1.3)
        else:
            return self._apply_clahe(image)

    def denoise_image(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Remove noise from image.

        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median', 'nlmeans')

        Returns:
            Denoised image
        """
        try:
            if method == 'bilateral':
                return cv2.bilateralFilter(image, *self.bilateral_filter_params)
            elif method == 'gaussian':
                return cv2.GaussianBlur(image, self.gaussian_blur_size, 0)
            elif method == 'median':
                return cv2.medianBlur(image, 5)
            elif method == 'nlmeans':
                if len(image.shape) == 3:
                    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                else:
                    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            else:
                logger.warning(f"Unknown denoising method: {method}")
                return image
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image

    def enhance_sharpness(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Enhance image sharpness.

        Args:
            image: Input image
            strength: Sharpening strength (0.0 to 2.0)

        Returns:
            Sharpened image
        """
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)

            # Blend with original based on strength
            return cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image

    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image values.

        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore', 'robust')

        Returns:
            Normalized image
        """
        try:
            if method == 'minmax':
                # Min-max normalization to [0, 255]
                normalized = cv2.normalize(
                    image, None, 0, 255, cv2.NORM_MINMAX)
                return normalized.astype(np.uint8)
            elif method == 'zscore':
                # Z-score normalization
                mean = np.mean(image)
                std = np.std(image)
                normalized = (image - mean) / (std + 1e-8)
                # Scale to [0, 255]
                normalized = ((normalized + 3) / 6) * 255
                return np.clip(normalized, 0, 255).astype(np.uint8)
            elif method == 'robust':
                # Robust normalization using percentiles
                p2 = np.percentile(image, 2)
                p98 = np.percentile(image, 98)
                normalized = np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)
                return (normalized * 255).astype(np.uint8)
            else:
                logger.warning(f"Unknown normalization method: {method}")
                return image
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return image

    def super_resolution(self, image: np.ndarray, scale_factor: int = 2) -> Optional[np.ndarray]:
        """
        Apply super-resolution to enhance image quality.

        Args:
            image: Input image
            scale_factor: Upscaling factor

        Returns:
            Super-resolved image or None if failed
        """
        try:
            # Try to use OpenCV's DNN super-resolution
            try:
                # You would need to download these models
                sr = cv2.dnn_superres.DnnSuperResImpl_create()

                if scale_factor == 2:
                    sr.readModel("ESPCN_x2.pb")
                    sr.setModel("espcn", 2)
                elif scale_factor == 3:
                    sr.readModel("ESPCN_x3.pb")
                    sr.setModel("espcn", 3)
                elif scale_factor == 4:
                    sr.readModel("ESPCN_x4.pb")
                    sr.setModel("espcn", 4)
                else:
                    logger.warning(f"Unsupported scale factor: {scale_factor}")
                    return None

                return sr.upsample(image)

            except Exception:
                # Fallback to bicubic interpolation
                h, w = image.shape[:2]
                new_size = (w * scale_factor, h * scale_factor)
                return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        except Exception as e:
            logger.warning(f"Super-resolution failed: {e}")
            return None

    def preprocess_pipeline(self, image: np.ndarray, config: dict) -> np.ndarray:
        """
        Apply a complete preprocessing pipeline.

        Args:
            image: Input image
            config: Processing configuration

        Returns:
            Processed image
        """
        result = image.copy()

        try:
            # Apply denoising if enabled
            if config.get('denoise', False):
                method = config.get('denoise_method', 'bilateral')
                result = self.denoise_image(result, method)

            # Apply lighting enhancement
            if config.get('enhance_lighting', True):
                method = config.get('lighting_method', 'adaptive')
                result = self.enhance_lighting(result, method)

            # Apply sharpening if enabled
            if config.get('sharpen', False):
                strength = config.get('sharpen_strength', 1.0)
                result = self.enhance_sharpness(result, strength)

            # Apply super-resolution if enabled
            if config.get('super_resolution', False):
                scale = config.get('sr_scale_factor', 2)
                sr_result = self.super_resolution(result, scale)
                if sr_result is not None:
                    result = sr_result

            # Apply normalization
            if config.get('normalize', True):
                method = config.get('normalize_method', 'minmax')
                result = self.normalize_image(result, method)

            return result

        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            return image


def enhance_image(image: np.ndarray, config: Optional[dict] = None) -> np.ndarray:
    """
    Convenience function for image enhancement.

    Args:
        image: Input image
        config: Enhancement configuration

    Returns:
        Enhanced image
    """
    if config is None:
        config = {
            'enhance_lighting': True,
            'lighting_method': 'adaptive',
            'normalize': True,
            'normalize_method': 'minmax'
        }

    enhancer = ImageEnhancer()
    return enhancer.preprocess_pipeline(image, config)
