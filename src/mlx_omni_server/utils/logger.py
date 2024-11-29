import logging
import os
import sys
from logging.handlers import RotatingFileHandler

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

logger.propagate = False
