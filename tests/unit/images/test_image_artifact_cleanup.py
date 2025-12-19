import os
import time
from pathlib import Path

import pytest

from mlx_omni_server.optional_features import install_instructions, missing_packages

if missing_packages("images"):
    pytest.skip(
        f"Images extra is not installed; install with {install_instructions('images')}.",
        allow_module_level=True,
    )

from mlx_omni_server.images.images_service import cleanup_expired_url_images


def test_cleanup_expired_url_images(tmp_path: Path):
    new_file = tmp_path / "new.png"
    new_file.write_bytes(b"x")

    old_file = tmp_path / "old.png"
    old_file.write_bytes(b"x")
    old_mtime = time.time() - 60
    os.utime(old_file, (old_mtime, old_mtime))

    removed = cleanup_expired_url_images(tmp_path, ttl_seconds=5)

    assert removed == 1
    assert not old_file.exists()
    assert new_file.exists()
