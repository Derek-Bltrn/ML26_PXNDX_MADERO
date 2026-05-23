import logging
import os
from datetime import datetime
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
LOGS_DIR = CURRENT_FILE.parent / "logs"


def setup_logger(name: str, log_dir: Path = LOGS_DIR) -> logging.Logger:
    """
    Setup a logger that writes to console and to <log_dir>/<timestamp>_<name>.log.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
