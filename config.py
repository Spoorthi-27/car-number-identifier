"""
config.py
---------
Central place to tweak the pipeline without hunting through the code.
Paths are resolved from this file's directory so the database stays put
no matter which folder the server is started from.

Optional environment variables (see .env.example) override the defaults.
None of them are secrets.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate)


# --- Camera / video ---
CAMERA_INDEX = _env_int("CAMERA_INDEX", 0)  # 0 = default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- YOLO plate detector ---
# Pretrained license-plate detection weights, auto-downloaded from Hugging Face
# on first run (via the `yolov5` package). No training required to get started.
MODEL_NAME = os.getenv("MODEL_NAME", "keremberke/yolov5s-license-plate")
CONF_THRESHOLD = _env_float("CONF_THRESHOLD", 0.35)
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 10
PROCESS_EVERY_N_FRAMES = 3  # live feed: run YOLO+OCR on 1 of every N frames

# --- OCR / plate-text filtering ---
MIN_PLATE_TEXT_LENGTH = 4
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or None

# --- Database / logging ---
DB_PATH = _resolve(os.getenv("DB_PATH", "database/plates.db"))
DEDUPE_SECONDS = _env_int("DEDUPE_SECONDS", 5)
SAVE_SNAPSHOTS = os.getenv("SAVE_SNAPSHOTS", "1").strip().lower() not in {"0", "false", "no"}
SNAPSHOT_DIR = _resolve(os.getenv("SNAPSHOT_DIR", "output/snapshots"))

# Uploaded video: sample frames instead of running OCR on every frame.
VIDEO_SAMPLE_EVERY = _env_int("VIDEO_SAMPLE_EVERY", 10)
VIDEO_MAX_FRAMES = _env_int("VIDEO_MAX_FRAMES", 40)
