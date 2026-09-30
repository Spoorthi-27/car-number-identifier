"""
utils.py
--------
Small helper functions shared across the CLI, the web app, and the tests.
"""
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array (OpenCV convention)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{path}'. Check the path and file format.")
    return image


def save_image(path: str, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def make_synthetic_test_image(
    plate_text: str = "MH12AB1234",
    out_path: str = None,
) -> np.ndarray:
    """
    Generate a synthetic 'car photo' with a rectangular white plate and
    black text on it. Useful for smoke-testing the pipeline without
    needing a real photograph on hand.
    """
    img = np.full((480, 640, 3), (90, 60, 40), dtype=np.uint8)  # dark bluish "car body"

    # Draw a plate-like white rectangle
    px, py, pw, ph = 190, 300, 260, 70
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 0, 0), 3)

    cv2.putText(
        img,
        plate_text,
        (px + 15, py + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    if out_path:
        save_image(out_path, img)

    return img
