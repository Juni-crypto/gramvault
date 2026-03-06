"""Structured logging setup."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from config import Config


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("instaintel")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s.%(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File (rotating: 10MB max, 5 backups)
        Config.ensure_dirs()
        fh = RotatingFileHandler(
            Config.DATA_DIR / "instaintel.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


log = setup_logging()
