"""
pipeline.py
-----------
Wires the YOLO detector + OCR together into a single end-to-end pipeline:
    frame -> YOLO detect plate boxes -> crop -> preprocess -> OCR -> annotated result
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .yolo_detector import YoloPlateDetector
from .ocr import PlateOCR
from .district import extract_valid_plate


@dataclass
class PlateResult:
    box: tuple           # (x, y, w, h)
    text: str            # the extracted/cleaned plate text if one was found, else the raw OCR text
    detection_conf: float   # YOLO's confidence that this box is a plate
    confidence_ok: bool      # heuristic: was a clean, known-state plate-shaped substring found in the OCR text?


class PlateReaderPipeline:
    def __init__(
        self,
        model_name: str = "keremberke/yolov5s-license-plate",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        tesseract_cmd: str = None,
        min_plate_text_length: int = 4,
    ):
        self.detector = YoloPlateDetector(model_name, conf_threshold, iou_threshold)
        self.ocr = PlateOCR(tesseract_cmd)
        self.min_plate_text_length = min_plate_text_length

    def process(self, image: np.ndarray) -> List[PlateResult]:
        boxes = self.detector.detect(image)
        results: List[PlateResult] = []

        for (x, y, w, h, conf) in boxes:
            pad_x, pad_y = int(0.05 * w), int(0.15 * h)
            x1, y1 = max(x - pad_x, 0), max(y - pad_y, 0)
            x2, y2 = min(x + w + pad_x, image.shape[1]), min(y + h + pad_y, image.shape[0])

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            raw_text = self.ocr.read_plate(crop)
            cleaned = extract_valid_plate(raw_text)
            text = cleaned if cleaned else raw_text
            confidence_ok = cleaned is not None

            results.append(
                PlateResult(
                    box=(x1, y1, x2 - x1, y2 - y1),
                    text=text,
                    detection_conf=conf,
                    confidence_ok=confidence_ok,
                )
            )

        return results

    def annotate(self, image: np.ndarray, results: List[PlateResult]) -> np.ndarray:
        annotated = image.copy()
        for r in results:
            x, y, w, h = r.box
            color = (0, 200, 0) if r.confidence_ok else (0, 165, 255)  # green / orange (BGR)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{r.text or '?'} ({r.detection_conf:.2f})"
            cv2.putText(
                annotated,
                label,
                (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated
