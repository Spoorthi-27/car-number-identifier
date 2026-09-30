"""
logging_setup.py
----------------
One application logger for the plate reader. Call get_logger() from any
module; the first call attaches a single console handler.
"""
import logging

_LOGGER_NAME = "plate_reader"
_CONFIGURED = False


def setup_logging() -> logging.Logger:
    """Attach a console handler once. Safe to call from every entry point."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED or logger.handlers:
        _CONFIGURED = True
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger() -> logging.Logger:
    return setup_logging()
