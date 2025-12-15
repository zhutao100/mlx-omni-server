import uvicorn

from mlx_omni_server.utils.logger import configure_logging


def test_uvicorn_accepts_custom_log_config(tmp_path):
    log_config = configure_logging(
        log_level="debug",
        log_file=True,
        log_dir=tmp_path,
        log_file_format="jsonl",
    )

    uvicorn.Config(
        "mlx_omni_server.main:app",
        log_level="debug",
        log_config=log_config,
        use_colors=True,
    )
