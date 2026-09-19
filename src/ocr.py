"""
ocr.py
------
Text *recognition*: given a cropped plate image (from the YOLO detector),
clean it up and run it through Tesseract OCR, then post-process the raw
text into a plausible plate string.

Preprocessing steps, in order:
  1. Grayscale
  2. Resize / upscale (small crops are hard for OCR to read accurately)
  3. Deskew (straighten plates photographed at an angle)
  4. Denoise (remove sensor/compression noise)
  5. Brightness & contrast normalization (CLAHE)
  6. Binarize (Otsu thresholding) so characters are crisp black-on-white

OCR itself tries a few different Tesseract page-segmentation strategies
(single line, single word, sparse text) and keeps whichever produces the
most plausible-length plate string, since no single mode is best for
every angle/plate style.
"""
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

PLATE_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Page-segmentation modes to try, in order of preference. 7 = single text
# line (best for a clean, level plate); 8 = single word; 11 = sparse text,
# no particular layout (most forgiving, best last resort for skewed/split text).
_PSM_MODES_TO_TRY = [7, 8, 11]

# Common Windows install locations, checked only if `tesseract` isn't already
# on PATH - avoids needing to manually set tesseract_cmd on most setups.
_WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _autodetect_tesseract() -> str:
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _WINDOWS_FALLBACK_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


class PlateOCR:
    def __init__(self, tesseract_cmd: str = None, whitelist: str = PLATE_WHITELIST):
        resolved_cmd = tesseract_cmd or _autodetect_tesseract()
        if resolved_cmd:
            pytesseract.pytesseract.tesseract_cmd = resolved_cmd
        else:
            print(
                "WARNING: Could not find the Tesseract executable automatically. "
                "If OCR fails, install Tesseract (see README) or pass its path "
                "explicitly: PlateOCR(tesseract_cmd=r'C:\\path\\to\\tesseract.exe')"
            )
        self.whitelist = whitelist

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Estimate the tilt of the plate (via the minimum-area rectangle around
        its largest bright region) and rotate the crop to straighten it.
        Skips rotation if no reliable region is found, or the tilt is already
        small enough not to matter.
        """
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        largest = max(contours, key=cv2.contourArea)
        image_area = gray.shape[0] * gray.shape[1]
        if cv2.contourArea(largest) < 0.08 * image_area:
            return gray  # too small/unreliable a region to trust for angle estimation

        angle = cv2.minAreaRect(largest)[-1]
        # cv2.minAreaRect reports angle in [-90, 0); normalize to roughly [-45, 45]
        if angle < -45:
            angle += 90
        if abs(angle) < 1.5:
            return gray  # already close enough to level

        h, w = gray.shape
        rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            gray, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img

        # 1. Upscale - plate crops from a YOLO box are often small (~100x30px)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        # 2. Straighten plates photographed at an angle
        gray = self._deskew(gray)

        # 3. Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # 4. Brightness/contrast normalization (handles glare, shadows, dusk shots)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 5. Binarize
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def read_plate(self, plate_img: np.ndarray) -> str:
        processed = self.preprocess(plate_img)

        best_text = ""
        for psm in _PSM_MODES_TO_TRY:
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={self.whitelist}"
            raw_text = pytesseract.image_to_string(processed, config=config)
            text = self._clean(raw_text)
            if len(text) > len(best_text):
                best_text = text
            if len(best_text) >= 6:  # good enough plate-length read, stop trying more modes
                break

        return best_text

    @staticmethod
    def _clean(text: str) -> str:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text.strip()
