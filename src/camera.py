"""
camera.py
---------
Open a webcam without crashing when the device is missing or permission
was not granted. The same helper is used by the API stream and the CLI.
"""
import os
import sys
import time
from typing import List, Optional

import cv2

from .logging_setup import get_logger

logger = get_logger()

CAMERA_UNAVAILABLE = "Unable to access camera. Please check camera permissions."


class CameraError(Exception):
    """Raised when a camera cannot be opened or cannot deliver a frame."""


class VideoOpenError(Exception):
    """Raised when a video file cannot be opened."""


def _candidate_backends() -> List[int]:
    backends = []
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    elif sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    elif sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    backends.append(cv2.CAP_ANY)
    return backends


def _try_open_once(camera_index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    for backend in _candidate_backends():
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                logger.info("Camera started (index %s)", camera_index)
                return cap
        cap.release()
    return None


def open_camera(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    attempts: int = 3,
    delay: float = 0.25,
) -> cv2.VideoCapture:
    """
    Open a webcam and read one frame so a permission failure is detected
    immediately. Raises CameraError with a user-facing message.
    """
    last_opened = None
    for attempt in range(max(1, attempts)):
        last_opened = _try_open_once(camera_index, width, height)
        if last_opened is not None:
            return last_opened
        if attempt + 1 < attempts:
            time.sleep(delay)

    logger.error("Camera error (index %s): %s", camera_index, CAMERA_UNAVAILABLE)
    raise CameraError(CAMERA_UNAVAILABLE)


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file. Raises VideoOpenError instead of crashing."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        message = f"Video cannot be opened: {path}"
        logger.error(message)
        raise VideoOpenError(message)
    return cap


def release_capture(cap: Optional[cv2.VideoCapture]) -> None:
    if cap is None:
        return
    try:
        cap.release()
    except Exception as exc:
        logger.error("Failed to release camera: %s", exc)
