"""Plate-color classification uses only the plate crop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.plate_color import classify_plate_color


def _plate(background, text_color, text="KA09AB1234"):
    image = np.full((80, 280, 3), background, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (279, 79), (0, 0, 0), 2)
    cv2.putText(image, text, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)
    return image


def test_white_yellow_and_green_plates():
    assert classify_plate_color(_plate((255, 255, 255), (0, 0, 0)))[0] == "WHITE"
    assert classify_plate_color(_plate((0, 220, 220), (0, 0, 0)))[0] == "YELLOW"
    assert classify_plate_color(_plate((70, 180, 50), (255, 255, 255)))[0] == "GREEN"


def test_white_plate_on_a_green_vehicle_stays_white():
    plate = _plate((255, 255, 255), (0, 0, 0))
    vehicle = np.full((220, 420, 3), (40, 160, 40), dtype=np.uint8)
    vehicle[70:150, 70:350] = plate
    # A loose crop still includes green body around the plate.
    loose = vehicle[45:175, 40:380]
    color, _confidence = classify_plate_color(loose)
    assert color == "WHITE"


def test_unreadable_color_is_unknown_and_does_not_crash():
    red = _plate((30, 30, 210), (255, 255, 255))
    assert classify_plate_color(red)[0] == "UNKNOWN"
    assert classify_plate_color(np.zeros((4, 4, 3), dtype=np.uint8))[0] == "UNKNOWN"
    assert classify_plate_color(None) == ("UNKNOWN", 0.0)


def test_tilted_yellow_plate():
    plate = _plate((0, 210, 220), (0, 0, 0))
    height, width = plate.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 12, 1.0)
    tilted = cv2.warpAffine(plate, matrix, (width, height), borderValue=(20, 20, 20))
    assert classify_plate_color(tilted)[0] == "YELLOW"
