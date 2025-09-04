"""
Caching Module
Intelligent caching system for improved performance.
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import pickle
import time
import os
import threading
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class IntelligentCache:
    """Intelligent caching system for face detection results."""

    def __init__(self, max_memory_items: int = 1000, max_disk_size_mb: int = 100, cache_dir: str = "cache"):
        self.max_memory_items = max_memory_items
        self.max_disk_size_mb = max_disk_size_mb
        self.cache_dir = cache_dir

        # Memory cache (LRU)
        self.memory_cache = OrderedDict()
        self.memory_stats = {'hits': 0, 'misses': 0}

        # Disk cache
        self.disk_cache_index = {}
        self.disk_stats = {'hits': 0, 'misses': 0, 'size_mb': 0}

        # Thread safety
        self.lock = threading.RLock()

        # Initialize
        self._setup_cache_directory()
        self._load_disk_index()

    def _setup_cache_directory(self):
        """Setup cache directory structure."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir,
                        'detection'), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir,
                        'recognition'), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir,
                        'embeddings'), exist_ok=True)
        except Exception as e:
            logger.error(f"Cache directory setup failed: {e}")

    def _load_disk_index(self):
        """Load disk cache index."""
        try:
            index_file = os.path.join(self.cache_dir, 'index.pkl')
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    self.disk_cache_index = pickle.load(f)

                # Calculate current disk usage
                total_size = 0
                for cache_info in self.disk_cache_index.values():
                    file_path = cache_info['file_path']
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)

                self.disk_stats['size_mb'] = total_size / (1024 * 1024)
                logger.info(
                    f"Loaded disk cache: {len(self.disk_cache_index)} items, {self.disk_stats['size_mb']:.2f} MB")
        except Exception as e:
            logger.error(f"Disk index loading failed: {e}")
            self.disk_cache_index = {}

    def _save_disk_index(self):
        """Save disk cache index."""
        try:
            index_file = os.path.join(self.cache_dir, 'index.pkl')
            with open(index_file, 'wb') as f:
                pickle.dump(self.disk_cache_index, f)
        except Exception as e:
            logger.error(f"Disk index saving failed: {e}")

    def _generate_image_hash(self, image: np.ndarray) -> str:
        """Generate hash for image content."""
        try:
            # Use image content and shape for hashing
            content_hash = hashlib.md5(image.tobytes()).hexdigest()
            shape_info = f"{image.shape}_{image.dtype}"
            return hashlib.md5((content_hash + shape_info).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Image hashing failed: {e}")
            return str(time.time())  # Fallback to timestamp

    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration."""
        try:
            config_str = str(sorted(config.items()))
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Config hashing failed: {e}")
            return ""

    def get_detection_cache_key(self, image: np.ndarray, detector_name: str, config: Dict[str, Any]) -> str:
        """Generate cache key for detection results."""
        image_hash = self._generate_image_hash(image)
        config_hash = self._generate_config_hash(config)
        return f"detection_{detector_name}_{image_hash}_{config_hash}"

    def get_recognition_cache_key(self, face_embedding: np.ndarray, database_version: str) -> str:
        """Generate cache key for recognition results."""
        embedding_hash = hashlib.md5(face_embedding.tobytes()).hexdigest()
        return f"recognition_{embedding_hash}_{database_version}"

    def get_embedding_cache_key(self, face_image: np.ndarray, model_name: str) -> str:
        """Generate cache key for face embeddings."""
        image_hash = self._generate_image_hash(face_image)
        return f"embedding_{model_name}_{image_hash}"

    def get(self, cache_key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                # Move to end (most recently used)
                value = self.memory_cache.pop(cache_key)
                self.memory_cache[cache_key] = value
                self.memory_stats['hits'] += 1
                return value

            # Check disk cache
            if cache_key in self.disk_cache_index:
                try:
                    cache_info = self.disk_cache_index[cache_key]
                    file_path = cache_info['file_path']

                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            value = pickle.load(f)

                        # Update access time
                        cache_info['last_access'] = time.time()

                        # Add to memory cache
                        self._add_to_memory_cache(cache_key, value)

                        self.disk_stats['hits'] += 1
                        return value
                    else:
                        # File doesn't exist, remove from index
                        del self.disk_cache_index[cache_key]
                        self._save_disk_index()
                except Exception as e:
                    logger.error(f"Disk cache read failed: {e}")

            # Cache miss
            self.memory_stats['misses'] += 1
            self.disk_stats['misses'] += 1
            return None

    def set(self, cache_key: str, value: Any, cache_type: str = 'detection'):
        """Set item in cache."""
        with self.lock:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, value)

            # Add to disk cache for persistent storage
            self._add_to_disk_cache(cache_key, value, cache_type)

    def _add_to_memory_cache(self, cache_key: str, value: Any):
        """Add item to memory cache with LRU eviction."""
        try:
            # Remove if already exists
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]

            # Add new item
            self.memory_cache[cache_key] = value

            # Evict oldest items if necessary
            while len(self.memory_cache) > self.max_memory_items:
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]

        except Exception as e:
            logger.error(f"Memory cache addition failed: {e}")

    def _add_to_disk_cache(self, cache_key: str, value: Any, cache_type: str):
        """Add item to disk cache."""
        try:
            # Create file path
            file_name = f"{cache_key}.pkl"
            file_path = os.path.join(self.cache_dir, cache_type, file_name)

            # Save to disk
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)

            # Update index
            file_size = os.path.getsize(file_path)
            self.disk_cache_index[cache_key] = {
                'file_path': file_path,
                'cache_type': cache_type,
                'created': time.time(),
                'last_access': time.time(),
                'size': file_size
            }

            # Update disk usage
            self.disk_stats['size_mb'] += file_size / (1024 * 1024)

            # Clean up if necessary
            self._cleanup_disk_cache()

            # Save index
            self._save_disk_index()

        except Exception as e:
            logger.error(f"Disk cache addition failed: {e}")

    def _cleanup_disk_cache(self):
        """Clean up disk cache when size limit is exceeded."""
        try:
            if self.disk_stats['size_mb'] <= self.max_disk_size_mb:
                return

            # Sort by last access time (oldest first)
            sorted_items = sorted(
                self.disk_cache_index.items(),
                key=lambda x: x[1]['last_access']
            )

            # Remove oldest items until under limit
            for cache_key, cache_info in sorted_items:
                if self.disk_stats['size_mb'] <= self.max_disk_size_mb:
                    break

                try:
                    file_path = cache_info['file_path']
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.disk_stats['size_mb'] -= file_size / (1024 * 1024)

                    del self.disk_cache_index[cache_key]

                except Exception as e:
                    logger.error(f"Cache cleanup failed for {cache_key}: {e}")

        except Exception as e:
            logger.error(f"Disk cache cleanup failed: {e}")

    def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern."""
        with self.lock:
            try:
                if pattern is None:
                    # Clear all caches
                    self.memory_cache.clear()

                    # Clear disk cache
                    for cache_info in self.disk_cache_index.values():
                        try:
                            if os.path.exists(cache_info['file_path']):
                                os.remove(cache_info['file_path'])
                        except Exception as e:
                            logger.error(f"Cache file removal failed: {e}")

                    self.disk_cache_index.clear()
                    self.disk_stats['size_mb'] = 0
                    self._save_disk_index()
                else:
                    # Clear entries matching pattern
                    memory_keys_to_remove = [
                        k for k in self.memory_cache.keys() if pattern in k]
                    for key in memory_keys_to_remove:
                        del self.memory_cache[key]

                    disk_keys_to_remove = [
                        k for k in self.disk_cache_index.keys() if pattern in k]
                    for key in disk_keys_to_remove:
                        cache_info = self.disk_cache_index[key]
                        try:
                            if os.path.exists(cache_info['file_path']):
                                file_size = os.path.getsize(
                                    cache_info['file_path'])
                                os.remove(cache_info['file_path'])
                                self.disk_stats['size_mb'] -= file_size / \
                                    (1024 * 1024)
                        except Exception as e:
                            logger.error(f"Cache file removal failed: {e}")

                        del self.disk_cache_index[key]

                    if disk_keys_to_remove:
                        self._save_disk_index()

                logger.info(f"Cache invalidated: pattern='{pattern}'")

            except Exception as e:
                logger.error(f"Cache invalidation failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            memory_hit_rate = self.memory_stats['hits'] / max(
                1, self.memory_stats['hits'] + self.memory_stats['misses'])
            disk_hit_rate = self.disk_stats['hits'] / max(
                1, self.disk_stats['hits'] + self.disk_stats['misses'])

            return {
                'memory_cache': {
                    'items': len(self.memory_cache),
                    'max_items': self.max_memory_items,
                    'hits': self.memory_stats['hits'],
                    'misses': self.memory_stats['misses'],
                    'hit_rate': memory_hit_rate
                },
                'disk_cache': {
                    'items': len(self.disk_cache_index),
                    'size_mb': self.disk_stats['size_mb'],
                    'max_size_mb': self.max_disk_size_mb,
                    'hits': self.disk_stats['hits'],
                    'misses': self.disk_stats['misses'],
                    'hit_rate': disk_hit_rate
                }
            }

    def precompute_cache(self, images: List[np.ndarray], detector, config: Dict[str, Any]):
        """Precompute cache for a batch of images."""
        logger.info(f"Precomputing cache for {len(images)} images")

        for i, image in enumerate(images):
            cache_key = self.get_detection_cache_key(
                image, detector.__class__.__name__, config)

            if self.get(cache_key) is None:
                try:
                    # Run detection and cache result
                    result = detector.detect_faces(image)
                    self.set(cache_key, result, 'detection')

                    if i % 10 == 0:
                        logger.info(f"Precomputed {i+1}/{len(images)} images")

                except Exception as e:
                    logger.error(f"Precomputation failed for image {i}: {e}")


class MemoryPool:
    """Memory pool for efficient memory management."""

    def __init__(self, initial_size: int = 100):
        self.pools = {}
        self.lock = threading.RLock()
        self.initial_size = initial_size

    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.uint8) -> np.ndarray:
        """Get array from pool or create new one."""
        with self.lock:
            pool_key = (shape, dtype)

            if pool_key not in self.pools:
                self.pools[pool_key] = []

            pool = self.pools[pool_key]

            if pool:
                return pool.pop()
            else:
                return np.zeros(shape, dtype=dtype)

    def return_array(self, array: np.ndarray):
        """Return array to pool for reuse."""
        with self.lock:
            pool_key = (array.shape, array.dtype)

            if pool_key not in self.pools:
                self.pools[pool_key] = []

            pool = self.pools[pool_key]

            # Limit pool size to prevent excessive memory usage
            if len(pool) < 50:
                # Clear array content
                array.fill(0)
                pool.append(array)

    def clear_pools(self):
        """Clear all memory pools."""
        with self.lock:
            self.pools.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            stats = {}
            total_arrays = 0

            for pool_key, pool in self.pools.items():
                shape, dtype = pool_key
                pool_size = len(pool)
                total_arrays += pool_size

                array_size_mb = np.prod(
                    shape) * np.dtype(dtype).itemsize / (1024 * 1024)
                total_size_mb = array_size_mb * pool_size

                stats[f"{shape}_{dtype}"] = {
                    'pool_size': pool_size,
                    'array_size_mb': array_size_mb,
                    'total_size_mb': total_size_mb
                }

            stats['summary'] = {
                'total_pools': len(self.pools),
                'total_arrays': total_arrays
            }

            return stats


def create_intelligent_cache(max_memory_items: int = 1000,
                             max_disk_size_mb: int = 100,
                             cache_dir: str = "cache") -> IntelligentCache:
    """Create an intelligent cache instance."""
    return IntelligentCache(max_memory_items, max_disk_size_mb, cache_dir)


def create_memory_pool(initial_size: int = 100) -> MemoryPool:
    """Create a memory pool instance."""
    return MemoryPool(initial_size)
