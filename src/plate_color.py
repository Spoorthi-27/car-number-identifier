"""
plate_color.py
--------------
Classify the background colour of a cropped number plate.

Only the plate crop is analysed. The surrounding vehicle is never sampled,
and dark glyphs (the characters) are masked out so black, white, or yellow
lettering does not decide the class.

Returns one of WHITE, YELLOW, GREEN, or UNKNOWN.
"""
from typing import Tuple

import cv2
import numpy as np

from .logging_setup import get_logger

logger = get_logger()

PLATE_COLORS = ("WHITE", "YELLOW", "GREEN", "UNKNOWN")


def classify_plate_color(plate_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Return (color, confidence) for a BGR plate crop.

    Confidence is the fraction of bright background pixels that voted for
    the winning colour, in the range 0..1. UNKNOWN is returned when the
    crop is empty, too small, or no colour clears the threshold.
    """
    try:
        roi = _plate_background_roi(plate_bgr)
        if roi is None:
            return "UNKNOWN", 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        # Characters and the dark border sit well below the reflective plate
        # background. Dropping them keeps black text on a yellow plate from
        # pulling the result toward black, and white text on a green plate
        # is a minority of the remaining pixels.
        background = val >= 110
        count = int(background.sum())
        if count < 25:
            return "UNKNOWN", 0.0

        hue_bg = hue[background]
        sat_bg = sat[background]
        val_bg = val[background]

        white = (sat_bg <= 48) & (val_bg >= 125)
        yellow = (hue_bg >= 15) & (hue_bg <= 36) & (sat_bg >= 70) & (val_bg >= 90)
        green = (hue_bg >= 38) & (hue_bg <= 95) & (sat_bg >= 45) & (val_bg >= 90)

        white_frac = float(white.mean())
        yellow_frac = float(yellow.mean())
        green_frac = float(green.mean())
        median_sat = float(np.median(sat_bg))
        median_val = float(np.median(val_bg))

        # A white plate under warm indoor light picks up a little hue, but
        # its saturation stays low. Require a truly saturated field before
        # calling the plate yellow or green.
        if median_sat <= 42 and median_val >= 140 and white_frac >= 0.50:
            if yellow_frac < 0.28 and green_frac < 0.28:
                return "WHITE", _confidence(white_frac)

        scores = {
            "WHITE": white_frac if median_sat <= 55 else white_frac * 0.45,
            "YELLOW": yellow_frac,
            "GREEN": green_frac,
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1]

        if best_score < 0.38:
            return "UNKNOWN", _confidence(best_score)
        if (best_score - second_score) < 0.10 and best_score < 0.62:
            return "UNKNOWN", _confidence(best_score)
        return best_name, _confidence(best_score)
    except Exception as exc:
        logger.error("Plate color detection failed: %s", exc)
        return "UNKNOWN", 0.0


def _plate_background_roi(plate_bgr: np.ndarray):
    """Inset the detection box so the vehicle around the plate is left out."""
    if plate_bgr is None or not isinstance(plate_bgr, np.ndarray) or plate_bgr.size == 0:
        return None
    if plate_bgr.ndim == 2:
        plate_bgr = cv2.cvtColor(plate_bgr, cv2.COLOR_GRAY2BGR)
    if plate_bgr.ndim != 3 or plate_bgr.shape[2] < 3:
        return None

    height, width = plate_bgr.shape[:2]
    if height < 8 or width < 16:
        return None

    # Skip the blue IND strip on the left and the outer border, which often
    # still contains a sliver of the vehicle when the detector box is loose.
    y0, y1 = int(height * 0.22), int(height * 0.80)
    x0, x1 = int(width * 0.18), int(width * 0.94)
    if y1 - y0 < 4 or x1 - x0 < 8:
        return plate_bgr
    return plate_bgr[y0:y1, x0:x1]


def _confidence(fraction: float) -> float:
    return round(max(0.0, min(0.99, fraction)), 3)
