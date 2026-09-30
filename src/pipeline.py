"""
pipeline.py
-----------
Wires detection, plate-color classification, OCR, and RTO lookup:

    frame -> detect plate -> crop plate -> color -> OCR -> normalize -> district
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .district import UNKNOWN_DISTRICT, lookup_registration
from .logging_setup import get_logger
from .ocr import PlateOCR
from .plate_color import classify_plate_color
from .vehicle_type import UNKNOWN_VEHICLE_TYPE, classify_vehicle_type
from .yolo_detector import YoloPlateDetector

logger = get_logger()


@dataclass
class PlateResult:
    box: tuple           # (x, y, w, h)
    text: str            # normalized plate when one was found, otherwise the raw OCR text
    detection_conf: float
    confidence_ok: bool
    raw_text: str = ""
    normalized_text: Optional[str] = None
    plate_color: str = "UNKNOWN"
    color_confidence: float = 0.0
    vehicle_type: str = UNKNOWN_VEHICLE_TYPE
    state: Optional[str] = None
    state_code: Optional[str] = None
    rto_code: Optional[str] = None
    rto_name: Optional[str] = None
    district: str = UNKNOWN_DISTRICT
    ocr_confidence: Optional[float] = None
    validation_note: Optional[str] = None
    ocr_failed: bool = False

    @property
    def status(self) -> str:
        """VALID, OCR_UNCERTAIN, or OCR_FAILED. The detection is kept either way."""
        if self.ocr_failed:
            return "OCR_FAILED"
        return "VALID" if self.confidence_ok else "OCR_UNCERTAIN"


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

        # A close-up plate needs a border added before the detector recognizes it as
    # a normal vehicle photo, but how much border varies by how tight the crop
    # is. A couple of ratios are tried, in order, stopping at the first hit.
    _PADDED_RETRY_RATIOS = (0.5, 0.9, 0.25)

    def _detect_on_padded(self, image: np.ndarray, pad_ratio: float) -> list:
        """Retry on a bordered copy so a close-up plate becomes a smaller part of the frame."""
        h, w = image.shape[:2]
        pad = int(pad_ratio * max(h, w))
        padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
        try:
            boxes = self.detector.detect(padded)
        except Exception as exc:
            logger.error("Padded plate detection failed: %s", exc)
            return []
        shifted = []
        for (x, y, bw, bh, conf) in boxes:
            x1 = max(int(x) - pad, 0)
            y1 = max(int(y) - pad, 0)
            x2 = min(int(x) + int(bw) - pad, w)
            y2 = min(int(y) + int(bh) - pad, h)
            if x2 > x1 and y2 > y1:
                shifted.append((x1, y1, x2 - x1, y2 - y1, conf))
        return shifted

    def process(self, image: np.ndarray) -> List[PlateResult]:
        try:
            boxes = self.detector.detect(image)
            if not boxes:
                for pad_ratio in self._PADDED_RETRY_RATIOS:
                    boxes = self._detect_on_padded(image, pad_ratio)
                    if boxes:
                        break
        except Exception as exc:

    def _read_text(self, tight: np.ndarray, padded: np.ndarray) -> tuple:
        """
        Read the detector box first. A padded crop includes vehicle body and
        can wash out yellow or green plates, so it is only a fallback.
        """
        raw_text, ocr_confidence = self.ocr.read_plate_details(tight)
        if lookup_registration(raw_text).normalized:
            return raw_text, ocr_confidence
        if padded is tight or padded.size == 0:
            return raw_text, ocr_confidence
        padded_text, padded_conf = self.ocr.read_plate_details(padded)
        padded_info = lookup_registration(padded_text)
        if padded_info.normalized and not lookup_registration(raw_text).normalized:
            return padded_text, padded_conf
        if len(padded_text) > len(raw_text):
            return padded_text, padded_conf
        return raw_text, ocr_confidence

    def _read_box(self, image: np.ndarray, x: int, y: int, w: int, h: int, conf: float) -> Optional[PlateResult]:
        height, width = image.shape[:2]
        x = max(int(x), 0)
        y = max(int(y), 0)
        w = max(int(w), 0)
        h = max(int(h), 0)
        if w == 0 or h == 0:
            logger.warning("Skipped a detection because the plate box was empty")
            return None

        # Color uses the detector box itself, inset inside classify_plate_color,
        # so the vehicle around the plate is not sampled. OCR keeps a little
        # padding because Tesseract reads clipped characters poorly.
        tight = image[y:min(y + h, height), x:min(x + w, width)]
        pad_x, pad_y = int(0.05 * w), int(0.15 * h)
        x1, y1 = max(x - pad_x, 0), max(y - pad_y, 0)
        x2, y2 = min(x + w + pad_x, width), min(y + h + pad_y, height)
        ocr_crop = image[y1:y2, x1:x2]
        if ocr_crop.size == 0:
            logger.warning("Skipped a detection because the plate crop was empty")
            return None

        plate_color, color_confidence = classify_plate_color(tight if tight.size else ocr_crop)
        raw_text, ocr_confidence = self._read_text(tight if tight.size else ocr_crop, ocr_crop)
        info = lookup_registration(raw_text)

        normalized = info.normalized
        text = normalized or raw_text or ""
        confidence_ok = normalized is not None
        ocr_failed = not raw_text
        note = None
        if ocr_failed:
            note = "OCR returned no text"
            logger.warning("OCR returned no text for a detected plate")
        elif not confidence_ok:
            note = "OCR text did not match a vehicle number pattern"
            logger.warning("OCR text %r did not match a vehicle number pattern", raw_text)
        elif len(normalized) < self.min_plate_text_length:
            confidence_ok = False
            note = "Plate text is shorter than the minimum length"

        return PlateResult(
            box=(x1, y1, x2 - x1, y2 - y1),
            text=text,
            detection_conf=float(conf),
            confidence_ok=confidence_ok,
            raw_text=raw_text or "",
            normalized_text=normalized,
            plate_color=plate_color,
            color_confidence=color_confidence,
            vehicle_type=classify_vehicle_type(plate_color),
            state=info.state,
            state_code=info.state_code,
            rto_code=info.rto_code,
            rto_name=info.rto_name,
            district=info.district,
            ocr_confidence=ocr_confidence,
            validation_note=note,
            ocr_failed=ocr_failed,
        )

    def annotate(self, image: np.ndarray, results: List[PlateResult]) -> np.ndarray:
        annotated = image.copy()
        for result in results:
            x, y, w, h = result.box
            color = (0, 200, 0) if result.confidence_ok else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = (
                f"{result.text or '?'} | {result.district} | {result.plate_color} "
                f"({result.detection_conf:.2f})"
            )
            cv2.putText(
                annotated,
                label,
                (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated