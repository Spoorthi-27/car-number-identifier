"""
records.py
----------
Save one processed plate to SQLite and write the application log lines
that describe that save. Callers decide whether a repeat webcam sighting
should be skipped; this module does the write itself.
"""
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2

import config
from .database import PlateDatabase
from .logging_setup import get_logger

logger = get_logger()

_dedupe_lock = threading.Lock()
_last_logged = {}


def is_recent_duplicate(plate_text: str, window_seconds: float) -> bool:
    """True when this plate was saved inside the webcam dedupe window."""
    key = (plate_text or "UNREADABLE").upper()
    now = time.time()
    with _dedupe_lock:
        previous = _last_logged.get(key, 0.0)
        return (now - previous) < window_seconds


def mark_logged(plate_text: str) -> None:
    key = (plate_text or "UNREADABLE").upper()
    with _dedupe_lock:
        _last_logged[key] = time.time()


def _safe_name(plate_text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", plate_text or "")
    return cleaned or "plate"


def _write_snapshot(frame, box, plate_text: str) -> Optional[str]:
    if frame is None or not box:
        return None
    x, y, width, height = [int(v) for v in box]
    x = max(x, 0)
    y = max(y, 0)
    crop = frame[y:y + height, x:x + width]
    if crop.size == 0:
        return None
    directory = Path(config.SNAPSHOT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_name(plate_text)}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    path = directory / filename
    if not cv2.imwrite(str(path), crop):
        logger.error("Failed to write plate snapshot %s", path.name)
        return None
    return str(path)


def save_plate_read(db: PlateDatabase, result, source: str, frame=None) -> dict:
    """
    Persist a pipeline result. Database errors are logged and returned;
    they are not raised, so a bad insert cannot take down the camera loop.
    """
    plate_text = result.text or ""
    logger.info("Plate detected: %s", plate_text or "(unreadable)")
    logger.info("OCR result: %s", result.raw_text or "")
    logger.info("Normalized plate: %s", result.normalized_text or "(none)")
    logger.info("RTO: %s", result.rto_code or "-")
    logger.info("District: %s", result.district)
    logger.info("Plate Color: %s", result.plate_color)
    if result.validation_note and not result.confidence_ok:
        logger.warning("Plate %s was not fully validated: %s", plate_text or "(unreadable)", result.validation_note)

    image_path = None
    if config.SAVE_SNAPSHOTS:
        try:
            image_path = _write_snapshot(frame, result.box, plate_text)
        except Exception as exc:
            logger.error("Failed to save plate snapshot: %s", exc)
            image_path = None

    try:
        row_id = db.log_plate(
            plate_text,
            confidence=result.detection_conf,
            image_path=image_path,
            status=result.status,
            plate_color=result.plate_color,
            vehicle_type=result.vehicle_type,
            state_name=result.state,
            state_code=result.state_code,
            rto_code=result.rto_code,
            rto_name=result.rto_name,
            district=result.district,
            ocr_confidence=result.ocr_confidence,
            ocr_raw=result.raw_text,
            source=source,
        )
        logger.info("Vehicle saved successfully")
        return {"saved": True, "id": row_id, "error": None, "image_path": image_path}
    except Exception as exc:
        logger.error("Failed to save vehicle %s: %s", plate_text or "(unreadable)", exc)
        return {"saved": False, "id": None, "error": str(exc), "image_path": image_path}
