import asyncio
import base64
import os
import random
import tempfile
import time
import uuid
from pathlib import Path

from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config import ModelConfig
from mflux.models.z_image.variants.turbo import ZImageTurbo
from mflux.utils.exceptions import StopImageGenerationException
from PIL import Image

from ..utils.logger import logger
from .schema import ImageGenerationRequest, ImageObject, ResponseFormat

IMAGE_URL_TTL_SECONDS = 60 * 60  # 1 hour
IMAGE_CLEANUP_INTERVAL_SECONDS = 10 * 60  # 10 minutes


def cleanup_expired_url_images(output_dir: Path, ttl_seconds: int = IMAGE_URL_TTL_SECONDS) -> int:
    """Delete old image artifacts left behind for `response_format=url`."""
    if not output_dir.exists():
        return 0

    now = time.time()
    removed = 0
    for image_path in output_dir.glob("*.png"):
        try:
            if now - image_path.stat().st_mtime > ttl_seconds:
                image_path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


async def background_url_image_cleanup(
    output_dir: Path,
    ttl_seconds: int = IMAGE_URL_TTL_SECONDS,
    interval_seconds: int = IMAGE_CLEANUP_INTERVAL_SECONDS,
) -> None:
    """Periodic cleanup task for URL-mode image artifacts."""
    from ..inference.runtime import run_blocking

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            removed = await run_blocking(cleanup_expired_url_images, output_dir, ttl_seconds)
            if removed:
                logger.debug("Cleaned up %d expired image artifacts", removed)
        except Exception:
            logger.exception("Error during image artifact cleanup")


class MFluxImageGenerator:
    """Image generator using mflux library"""

    def __init__(self, model_version: str = "filipstrand/Z-Image-Turbo-mflux-4bit"):
        self.model_version = model_version

        # Initialize model instance (lazy loading)
        self._model = None

    def _extra_base_model(self, model_name: str) -> str | None:
        # List of supported base models
        supported_base_models = ["schnell", "dev", "dev-fill", "dev-depth", "dev-redux"]
        base_model = None
        # Extract base_model from model_name if it contains any of the supported keywords
        model_name_lower = model_name.lower()
        for base in supported_base_models:
            if base in model_name_lower:
                base_model = base
                logger.info(
                    f"Extracted base_model '{base_model}' from model_name '{model_name}'"
                )
                break

        # If we couldn't extract a base_model, set it to None
        if not base_model:
            logger.info(
                f"Could not extract base_model from model_name '{model_name}', using None"
            )

        return base_model

    def _build_model(self, params: dict | None = None) -> ZImageTurbo:
        model_name = self.model_version
        base_model = params.get("base-model") if params else None
        if "/" in model_name and not base_model:
            base_model = self._extra_base_model(model_name)

        return ZImageTurbo(
            model_config=ModelConfig.from_name(
                model_name=model_name,
                base_model=base_model,  # type: ignore[arg-type]
            ),
            quantize=params.get("quantize") if params else None,
            model_path=params.get("local_path") if params else None,
            lora_paths=params.get("lora-paths") if params else None,
            lora_scales=params.get("lora-scales") if params else None,
        )

    def _get_model(self, params: dict | None = None) -> ZImageTurbo:
        """Get or initialize ZImageTurbo instance."""
        if self._model is None:
            self._model = self._build_model(params)
        return self._model

    def _parse_size(self, size_str: str) -> tuple[int, int]:
        """Parse size string to width and height"""
        try:
            width, height = map(int, size_str.split("x"))
            return width, height
        except (ValueError, AttributeError):
            return 1024, 1024

    def generate(
        self,
        request: ImageGenerationRequest,
        output_path: str,
        **extra_params,
    ) -> Image.Image:
        """Generate image using mflux"""
        # Parse image dimensions
        width, height = self._parse_size(request.size)

        # Get extra parameters from request
        request_extra_params = request.get_extra_params()

        # Merge all extra parameters, with passed extra_params taking precedence
        all_extra_params = {**request_extra_params, **extra_params}
        logger.info(f"all_extra_params: {all_extra_params}")

        low_ram = bool(
            all_extra_params.pop("low_ram", False)
            or all_extra_params.pop("low_memory_mode", False)
            or all_extra_params.pop("low_arm", False)
        )

        # Generate random seed if not specified
        seed = all_extra_params.pop("seed", random.randint(0, 2**32 - 1))

        # MemorySaver mutates the model by unloading encoders; don't reuse the model across calls.
        model = (
            self._build_model(all_extra_params) if low_ram else self._get_model(all_extra_params)
        )

        memory_saver = None
        if low_ram:
            memory_saver = MemorySaver(model=model, keep_transformer=False)
            model.callbacks.register(memory_saver)

        try:
            # Generate image
            image = model.generate_image(
                seed=seed,
                prompt=request.prompt,
                num_inference_steps=all_extra_params.pop("steps", 8),
                height=height,
                width=width,
            )

            # Save image
            image.save(output_path)

            return image
        except StopImageGenerationException as e:
            raise Exception(f"Image generation interrupted: {str(e)}")
        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")
        finally:
            if memory_saver:
                print(memory_saver.memory_stats())


class ImagesService:
    def __init__(self):
        # Use system temporary directory
        self.output_dir = Path(tempfile.gettempdir()) / "mlx_omni_server" / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache loaded generator instances
        self._generator_cache: dict[str, MFluxImageGenerator] = {}

    def _get_generator(self, model_name: str) -> MFluxImageGenerator:
        """Get or create image generator instance"""
        if model_name not in self._generator_cache:
            self._generator_cache[model_name] = MFluxImageGenerator(
                model_version=model_name
            )
        return self._generator_cache[model_name]

    def _get_output_path(self, uid: str) -> str:
        """Generate unique output path for image"""
        return str(self.output_dir / f"{uid}.png")

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _cleanup_image(self, image_path: str):
        """Clean up temporary image file"""
        try:
            os.unlink(image_path)
        except Exception as e:
            print(f"Error cleaning up image {image_path}: {str(e)}")

    def generate_images(
        self,
        request: ImageGenerationRequest,
    ) -> list[ImageObject]:
        """Generate images based on the request"""
        generated_images = []
        generator = self._get_generator(model_name=request.model)

        for _ in range(request.n):
            # Generate unique identifier for this image
            uid = uuid.uuid4().hex
            output_path = self._get_output_path(uid)

            try:
                # Generate the image
                generator.generate(request=request, output_path=output_path)

                # Create response object based on format
                image_object = ImageObject(revised_prompt=request.prompt)

                # Response format
                if request.response_format == ResponseFormat.B64_JSON:
                    image_object.b64_json = self._image_to_base64(output_path)
                else:  # URL format
                    image_object.url = f"file://{output_path}"

                generated_images.append(image_object)

            except Exception as e:
                raise Exception(f"Error generating image: {str(e)}")
            finally:
                # Clean up temporary file if using base64 format
                if request.response_format == ResponseFormat.B64_JSON:
                    self._cleanup_image(output_path)

        return generated_images
