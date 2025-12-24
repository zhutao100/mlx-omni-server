import asyncio
import base64
import io
import random
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, cast

from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config import ModelConfig
from mflux.models.z_image.variants.turbo import ZImageTurbo
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
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
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            return
        try:
            removed = await run_blocking(
                cleanup_expired_url_images,
                output_dir,
                ttl_seconds,
            )
            if removed:
                logger.debug("Cleaned up %d expired image artifacts", removed)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Error during image artifact cleanup")


class MFluxImageGenerator:
    """Image generator using mflux library"""

    def __init__(self, model_version: str = "filipstrand/Z-Image-Turbo-mflux-4bit"):
        self.model_version = model_version

        # Initialize model instance (lazy loading)
        self._model: ZImageTurbo | None = None
        self._model_init_signature: tuple[object, ...] | None = None

    @staticmethod
    def _get_first_param(params: dict, *names: str) -> object | None:
        for name in names:
            if name in params:
                return params[name]
        return None

    @staticmethod
    def _coerce_optional_str(value: object | None, *, field: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise ValueError(f"Invalid '{field}': expected a string")

    @staticmethod
    def _coerce_optional_int(value: object | None, *, field: str) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid '{field}': expected an integer") from exc

    @staticmethod
    def _coerce_optional_str_list(value: object | None, *, field: str) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list | tuple):
            items = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError(f"Invalid '{field}': expected a list of strings")
                items.append(item)
            return items or None
        raise ValueError(f"Invalid '{field}': expected a string or list of strings")

    @staticmethod
    def _coerce_optional_float_list(value: object | None, *, field: str) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, int | float):
            return [float(value)]
        if isinstance(value, list | tuple):
            items: list[float] = []
            for item in value:
                try:
                    items.append(float(item))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid '{field}': expected a list of numbers") from exc
            return items or None
        raise ValueError(f"Invalid '{field}': expected a number or list of numbers")

    def _normalize_model_init_params(
        self,
        params: dict | None,
    ) -> tuple[
        tuple[object, ...], str | None, int | None, str | None, list[str] | None, list[float] | None
    ]:
        params = params or {}
        model_name = self.model_version

        base_model = self._coerce_optional_str(
            self._get_first_param(params, "base_model", "base-model"),
            field="base_model",
        )
        quantize = self._coerce_optional_int(
            self._get_first_param(params, "quantize"),
            field="quantize",
        )
        model_path = self._coerce_optional_str(
            self._get_first_param(
                params,
                "model_path",
                "model-path",
                "local_path",
                "local-path",
            ),
            field="model_path",
        )
        if model_path is not None:
            model_path = str(Path(model_path).expanduser())

        lora_paths = self._coerce_optional_str_list(
            self._get_first_param(params, "lora_paths", "lora-paths"),
            field="lora_paths",
        )
        lora_scales = self._coerce_optional_float_list(
            self._get_first_param(params, "lora_scales", "lora-scales"),
            field="lora_scales",
        )
        if lora_paths is None:
            lora_scales = None

        signature = (
            model_name,
            base_model,
            quantize,
            model_path,
            tuple(lora_paths or ()),
            tuple(lora_scales or ()),
        )
        return signature, base_model, quantize, model_path, lora_paths, lora_scales

    def _build_model(
        self,
        base_model: str | None,
        quantize: int | None,
        model_path: str | None,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
    ) -> ZImageTurbo:
        model_config = ModelConfig.from_name(
            model_name=self.model_version,
            base_model=cast(Any, base_model),
        )
        return ZImageTurbo(
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def _get_model(self, params: dict | None = None) -> ZImageTurbo:
        """Get or initialize ZImageTurbo instance."""
        (
            signature,
            base_model,
            quantize,
            model_path,
            lora_paths,
            lora_scales,
        ) = self._normalize_model_init_params(params)
        if self._model is None or signature != self._model_init_signature:
            self._model = self._build_model(
                base_model=base_model,
                quantize=quantize,
                model_path=model_path,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            self._model_init_signature = signature
        return self._model

    def _parse_size(self, size_str: str | None) -> tuple[int, int]:
        """Parse size string to width and height"""
        if not size_str:
            return 1024, 1024
        try:
            width, height = map(int, size_str.split("x"))
            return width, height
        except (ValueError, AttributeError):
            return 1024, 1024

    def generate(
        self,
        request: ImageGenerationRequest,
        output_path: str | None = None,
        **extra_params,
    ) -> GeneratedImage:
        """Generate image using mflux"""
        # Parse image dimensions
        width, height = self._parse_size(request.size)

        # Get extra parameters from request
        request_extra_params = request.get_extra_params()

        # Merge all extra parameters, with passed extra_params taking precedence
        all_extra_params = {**request_extra_params, **extra_params}
        logger.debug("all_extra_params: %s", all_extra_params)

        low_ram = bool(
            all_extra_params.pop("low_ram", False)
            or all_extra_params.pop("low_memory_mode", False)
            or all_extra_params.pop("low_arm", False)
        )

        seed = self._coerce_optional_int(
            all_extra_params.pop("seed", None),
            field="seed",
        )
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        steps = self._coerce_optional_int(
            all_extra_params.pop("steps", 4),
            field="steps",
        )
        if steps is None or steps <= 0:
            raise ValueError("Invalid 'steps': expected a positive integer")

        scheduler = self._coerce_optional_str(
            all_extra_params.pop("scheduler", "linear"),
            field="scheduler",
        )
        if scheduler is None:
            scheduler = "linear"

        if low_ram:
            _, base_model, quantize, model_path, lora_paths, lora_scales = (
                self._normalize_model_init_params(all_extra_params)
            )
            model = self._build_model(
                base_model=base_model,
                quantize=quantize,
                model_path=model_path,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
        else:
            model = self._get_model(all_extra_params)

        memory_saver = None
        if low_ram:
            memory_saver = MemorySaver(model=model, keep_transformer=False)
            model.callbacks.register(memory_saver)

        try:
            generated = cast(
                GeneratedImage,
                model.generate_image(
                    seed=seed,
                    prompt=request.prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    scheduler=scheduler,
                ),
            )

            if output_path is not None:
                generated.save(path=output_path, export_json_metadata=False, overwrite=False)

            return generated
        except StopImageGenerationException as e:
            raise RuntimeError(f"Image generation interrupted: {e}") from e
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error generating image: {e}") from e
        finally:
            if memory_saver:
                logger.info("%s", memory_saver.memory_stats())


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

    def _image_to_base64_png(self, image: Image.Image) -> str:
        """Convert a PIL image to a base64-encoded PNG string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate_images(
        self,
        request: ImageGenerationRequest,
    ) -> list[ImageObject]:
        """Generate images based on the request"""
        generated_images = []
        model_name = request.model or "filipstrand/Z-Image-Turbo-mflux-4bit"
        generator = self._get_generator(model_name=model_name)
        response_format = request.response_format or ResponseFormat.B64_JSON

        for _ in range(request.n or 1):
            output_path = (
                self._get_output_path(uuid.uuid4().hex)
                if response_format == ResponseFormat.URL
                else None
            )

            generated = generator.generate(request=request, output_path=output_path)

            image_object = ImageObject(revised_prompt=request.prompt)
            if response_format == ResponseFormat.B64_JSON:
                image_object.b64_json = self._image_to_base64_png(generated.image)
            else:
                if output_path is None:
                    raise RuntimeError(
                        "Internal error: output_path is required for URL response format"
                    )
                image_object.url = f"file://{output_path}"

            generated_images.append(image_object)

        return generated_images
