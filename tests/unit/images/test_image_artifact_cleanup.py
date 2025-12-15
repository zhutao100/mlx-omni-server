import os
import time
from pathlib import Path

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
