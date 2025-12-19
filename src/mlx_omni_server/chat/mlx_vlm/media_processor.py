import asyncio
import base64
import gc
import hashlib
import os
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ...utils.logger import logger


class BaseProcessor(ABC):
    """Base class for media processors with common caching and session management."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        # Use tempfile for macOS-efficient temporary file handling
        self.temp_dir = tempfile.TemporaryDirectory()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_size = cache_size
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        # Replace lru_cache with manual cache for better control
        self._hash_cache: Dict[str, str] = {}
        self._cache_access_times: Dict[str, float] = {}

    def _get_media_hash(self, media_url: str) -> str:
        """Get hash for media URL with manual caching that can be cleared."""
        # Check if already cached
        if media_url in self._hash_cache:
            self._cache_access_times[media_url] = time.time()
            return self._hash_cache[media_url]

        # Generate hash
        if media_url.startswith("data:"):
            _, encoded = media_url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            data = media_url.encode('utf-8')

        hash_value = hashlib.md5(data).hexdigest()

        # Add to cache with size management
        if len(self._hash_cache) >= self._cache_size:
            self._evict_oldest_cache_entries()

        self._hash_cache[media_url] = hash_value
        self._cache_access_times[media_url] = time.time()
        return hash_value

    def _evict_oldest_cache_entries(self):
        """Remove oldest 20% of cache entries to make room."""
        if not self._cache_access_times:
            return

        # Sort by access time and remove oldest 20%
        sorted_items = sorted(self._cache_access_times.items(), key=lambda x: x[1])
        to_remove = len(sorted_items) // 5  # Remove 20%

        for url, _ in sorted_items[:to_remove]:
            self._hash_cache.pop(url, None)
            self._cache_access_times.pop(url, None)

        # Force garbage collection after cache eviction
        gc.collect()

    @abstractmethod
    def _get_media_format(self, media_url: str, data: bytes | None = None) -> str:
        """Determine media format from URL or data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _validate_media_data(self, data: bytes) -> bool:
        """Validate media data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _process_media_data(self, data: bytes, cached_path: str, **kwargs) -> str:
        """Process media data and save to cached path. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_media_type_name(self) -> str:
        """Get media type name for logging. Must be implemented by subclasses."""
        pass

    def _cleanup_old_files(self):
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                for file in os.listdir(self.temp_dir.name):
                    file_path = os.path.join(self.temp_dir.name, file)
                    if os.path.getmtime(file_path) < current_time - self._cleanup_interval:
                        os.remove(file_path)
                self._last_cleanup = current_time
                # Also clean up cache periodically
                if len(self._hash_cache) > self._cache_size * 0.8:
                    self._evict_oldest_cache_entries()
                gc.collect()  # Force garbage collection after cleanup
            except Exception as e:
                logger.warning(f"Failed to clean up old {self._get_media_type_name()} files: {str(e)}")

    async def _process_single_media(self, media_url: str, **kwargs) -> Tuple[str, str]:
        """Process a single media URL and return path to cached file and its hash."""
        try:
            media_hash = self._get_media_hash(media_url)
            media_format = self._get_media_format(media_url)
            cached_path = os.path.join(self.temp_dir.name, f"{media_hash}.{media_format}")

            if os.path.exists(cached_path):
                logger.debug(f"Using cached {self._get_media_type_name()}: {cached_path}")
                return cached_path, media_hash

            if os.path.exists(media_url):
                # Copy local file to cache
                with open(media_url, 'rb') as f:
                    data = f.read()

                if not self._validate_media_data(data):
                    raise ValueError(f"Invalid {self._get_media_type_name()} file format")

                result_path = self._process_media_data(data, cached_path, **kwargs)
                return result_path, media_hash

            elif media_url.startswith("data:"):
                _, encoded = media_url.split(",", 1)
                estimated_size = len(encoded) * 3 / 4
                if estimated_size > self._get_max_file_size():
                    raise ValueError(f"Base64-encoded {self._get_media_type_name()} exceeds size limit")
                data = base64.b64decode(encoded)

                if not self._validate_media_data(data):
                    raise ValueError(f"Invalid {self._get_media_type_name()} file format")

                result_path = self._process_media_data(data, cached_path, **kwargs)
                return result_path, media_hash
            else:
                # For URL-based media, we would need to implement HTTP fetching
                # For now, we'll raise an error since we don't have aiohttp dependency
                raise ValueError(f"URL-based media not supported without aiohttp dependency")

        except Exception as e:
            logger.error(f"Failed to process {self._get_media_type_name()}: {str(e)}")
            raise ValueError(f"Failed to process {self._get_media_type_name()}: {str(e)}")
        finally:
            gc.collect()

    def clear_cache(self):
        """Manually clear the hash cache to free memory."""
        self._hash_cache.clear()
        self._cache_access_times.clear()
        gc.collect()

    async def cleanup(self):
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        self._cleaned = True
        try:
            # Clear caches before cleanup
            self.clear_cache()
        except Exception as e:
            logger.warning(f"Exception during cleanup: {str(e)}")
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Exception shutting down executor: {str(e)}")
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Exception cleaning up temp directory: {str(e)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.cleanup()

    def __del__(self):
        # Async cleanup cannot be reliably performed in __del__
        # Please use 'async with Processor()' or call 'await cleanup()' explicitly.
        pass


class ImageProcessor(BaseProcessor):
    """Image processor for handling image files with caching, validation, and processing."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        super().__init__(max_workers, cache_size)
        Image.MAX_IMAGE_PIXELS = 100000000  # Limit to 100 megapixels

    def _get_media_format(self, media_url: str, data: bytes | None = None) -> str:
        """Determine image format from URL or data."""
        # For images, we always save as PNG for consistency
        return "png"

    def _validate_media_data(self, data: bytes) -> bool:
        """Basic validation of image data."""
        if len(data) < 100:  # Too small to be a valid image file
            return False

        # Check for common image file signatures
        image_signatures = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF87a
            b'GIF89a',  # GIF89a
            b'BM',  # BMP
            b'II*\x00',  # TIFF (little endian)
            b'MM\x00*',  # TIFF (big endian)
            b'RIFF',  # WebP (part of RIFF)
        ]

        for sig in image_signatures:
            if data.startswith(sig):
                return True

        # Additional check for WebP
        if data.startswith(b'RIFF') and b'WEBP' in data[:20]:
            return True

        return False

    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests."""
        return 30  # Standard timeout for images

    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return 100 * 1024 * 1024  # 100 MB limit for images

    def _get_media_type_name(self) -> str:
        """Get media type name for logging."""
        return "image"

    def _resize_image_keep_aspect_ratio(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image to {new_width}x{new_height} from {width}x{height}")

        return image

    def _prepare_image_for_saving(self, image: Image.Image) -> Image.Image:
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image, mask=image.split()[1])
            return background
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image

    def _process_media_data(self, data: bytes, cached_path: str, **kwargs) -> str:
        """Process image data and save to cached path."""
        image = None
        resize = kwargs.get("resize", True)
        try:
            with Image.open(BytesIO(data), mode='r') as image:
                if resize:
                    image = self._resize_image_keep_aspect_ratio(image)
                image = self._prepare_image_for_saving(image)
                image.save(cached_path, 'PNG', quality=100, optimize=True)

            self._cleanup_old_files()
            return cached_path
        finally:
            # Ensure image object is closed to free memory
            if image:
                try:
                    image.close()
                except:
                    pass

    async def process_image_url(self, image_url: str, resize: bool = True) -> Tuple[str, str]:
        """Process a single image URL and return path to cached file and its hash."""
        return await self._process_single_media(image_url, resize=resize)

    async def process_image_urls(self, image_urls: List[str], resize: bool = True) -> Tuple[List[str], List[str]]:
        """Process multiple image URLs and return paths to cached files and their hashes."""
        tasks = [self.process_image_url(url, resize=resize) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate paths and hashes
        paths = []
        hashes = []
        for result in results:
            if isinstance(result, tuple):
                path, hash_val = result
                paths.append(path)
                hashes.append(hash_val)
            else:
                # Handle exceptions
                logger.error(f"Error processing image: {result}")
                paths.append(None)
                hashes.append(None)

        # Force garbage collection after batch processing
        gc.collect()
        return paths, hashes


class AudioProcessor(BaseProcessor):
    """Audio processor for handling audio files with caching and validation."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        super().__init__(max_workers, cache_size)
        # Supported audio formats
        self._supported_formats = {'.mp3', '.wav'}

    def _get_media_format(self, media_url: str, data: bytes | None = None) -> str:
        """Determine audio format from URL or data."""
        if media_url.startswith("data:"):
            # Extract format from data URL
            mime_type = media_url.split(";")[0].split(":")[1]
            if "mp3" in mime_type or "mpeg" in mime_type:
                return "mp3"
            elif "wav" in mime_type:
                return "wav"
            elif "m4a" in mime_type or "mp4" in mime_type:
                return "m4a"
            elif "ogg" in mime_type:
                return "ogg"
            elif "flac" in mime_type:
                return "flac"
            elif "aac" in mime_type:
                return "aac"
        else:
            # Extract format from file extension
            ext = os.path.splitext(media_url.lower())[1]
            if ext in self._supported_formats:
                return ext[1:]  # Remove the dot

        # Default to wav if format cannot be determined
        return "wav"

    def _validate_media_data(self, data: bytes) -> bool:
        """Basic validation of audio data."""
        if len(data) < 100:  # Too small to be a valid audio file
            return False

        # Check for common audio file signatures
        audio_signatures = [
            b'ID3',  # MP3 with ID3 tag
            b'\xff\xfb',  # MP3 frame header
            b'\xff\xf3',  # MP3 frame header
            b'\xff\xf2',  # MP3 frame header
            b'RIFF',  # WAV/AVI
            b'OggS',  # OGG
            b'fLaC',  # FLAC
            b'\x00\x00\x00\x20ftypM4A',  # M4A
        ]

        for sig in audio_signatures:
            if data.startswith(sig):
                return True

        # Check for WAV format (RIFF header might be at different position)
        if b'WAVE' in data[:50]:
            return True

        return True  # Allow unknown formats to pass through

    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests."""
        return 60  # Longer timeout for audio files

    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return 500 * 1024 * 1024  # 500 MB limit for audio

    def _process_media_data(self, data: bytes, cached_path: str, **kwargs) -> str:
        """Process audio data and save to cached path."""
        with open(cached_path, 'wb') as f:
            f.write(data)
        self._cleanup_old_files()
        return cached_path

    def _get_media_type_name(self) -> str:
        """Get media type name for logging."""
        return "audio"

    async def process_audio_url(self, audio_url: str) -> Tuple[str, str]:
        """Process a single audio URL and return path to cached file and its hash."""
        return await self._process_single_media(audio_url)

    async def process_audio_urls(self, audio_urls: List[str]) -> Tuple[List[str], List[str]]:
        """Process multiple audio URLs and return paths to cached files and their hashes."""
        tasks = [self.process_audio_url(url) for url in audio_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate paths and hashes
        paths = []
        hashes = []
        for result in results:
            if isinstance(result, tuple):
                path, hash_val = result
                paths.append(path)
                hashes.append(hash_val)
            else:
                # Handle exceptions
                logger.error(f"Error processing audio: {result}")
                paths.append(None)
                hashes.append(None)

        # Force garbage collection after batch processing
        gc.collect()
        return paths, hashes


class MediaProcessor:
    """Unified processor for handling images and audio with caching"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.image_processor = ImageProcessor(max_workers=max_workers)
        self.audio_processor = AudioProcessor(max_workers=max_workers)

    def generate_media_hash(self, file_path: str) -> str:
        """Generate SHA256 hash for media file content."""
        if not file_path or not os.path.exists(file_path):
            return ""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    async def process_image_url(self, url: str, resize: bool = True) -> Tuple[str, str]:
        """Process a single image URL and return path to cached file and its hash"""
        return await self.image_processor.process_image_url(url, resize=resize)

    async def process_audio_url(self, url: str) -> Tuple[str, str]:
        """Process a single audio URL and return path to cached file and its hash"""
        return await self.audio_processor.process_audio_url(url)

    async def process_image_urls(self, urls: List[str], resize: bool = True) -> Tuple[List[str], List[str]]:
        """Process multiple image URLs concurrently and return paths and hashes"""
        return await self.image_processor.process_image_urls(urls, resize=resize)

    async def process_audio_urls(self, urls: List[str]) -> Tuple[List[str], List[str]]:
        """Process multiple audio URLs concurrently and return paths and hashes"""
        return await self.audio_processor.process_audio_urls(urls)

    async def cleanup(self):
        """Cleanup temporary files"""
        try:
            await self.image_processor.cleanup()
            await self.audio_processor.cleanup()
        except Exception as e:
            logger.error(f"Error during media processor cleanup: {e}")
