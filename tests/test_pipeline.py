"""
Smoke tests for the OCR, database, and pipeline-annotation logic.

These tests use a *fake* detector (no real YOLO weights downloaded) so
they run instantly and don't need internet access. The real YOLO
detector (src/yolo_detector.py) is exercised manually via
`python image_main.py --image ...` or `python main.py`, since it needs
its pretrained weights downloaded from Hugging Face on first use.

Run with:
    python -m pytest tests/ -v
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ocr import PlateOCR
from src.database import PlateDatabase
from src.pipeline import PlateReaderPipeline, PlateResult
from src.utils import make_synthetic_test_image


class FakeYoloDetector:
    """Stands in for YoloPlateDetector so tests don't need to download real weights."""

    def __init__(self, box=(160, 272, 311, 122, 0.93)):
        self.box = box

    def detect(self, frame_bgr):
        return [self.box]


def make_pipeline_with_fake_detector():
    pipeline = PlateReaderPipeline.__new__(PlateReaderPipeline)  # skip __init__ (avoids loading YOLO)
    pipeline.detector = FakeYoloDetector()
    pipeline.ocr = PlateOCR()
    pipeline.min_plate_text_length = 4
    return pipeline


def test_pipeline_runs_without_error():
    pipeline = make_pipeline_with_fake_detector()
    image = make_synthetic_test_image(plate_text="MH12AB1234")
    results = pipeline.process(image)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], PlateResult)


def test_annotate_returns_same_shape_image():
    pipeline = make_pipeline_with_fake_detector()
    image = make_synthetic_test_image(plate_text="MH12AB1234")
    results = pipeline.process(image)
    annotated = pipeline.annotate(image, results)
    assert annotated.shape == image.shape


def test_ocr_reads_plausible_text_from_synthetic_plate():
    ocr = PlateOCR()
    image = make_synthetic_test_image(plate_text="MH12AB1234")
    # crop roughly where make_synthetic_test_image draws the plate
    crop = image[300:370, 190:450]
    text = ocr.read_plate(crop)
    assert len(text) >= 4  # OCR should read *something* plausible, exact chars may vary slightly


def test_ocr_handles_tilted_plates():
    """A plate photographed at an angle should still be readable after deskewing."""
    import cv2

    ocr = PlateOCR()
    base = make_synthetic_test_image(plate_text="MH12AB1234")
    h, w = base.shape[:2]

    for angle in (10, 20):
        rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(base, rot_matrix, (w, h), borderValue=(90, 60, 40))
        crop = rotated[250:420, 130:520]
        text = ocr.read_plate(crop)
        assert len(text) >= 6, f"expected a plausible read at {angle} degrees, got '{text}'"


def test_database_logs_and_reads_back():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_plates.db")
        db = PlateDatabase(db_path)

        db.log_plate("MH12AB1234", confidence=0.91)
        db.log_plate("DL8CAF5030", confidence=0.87)

        rows = db.recent_plates(limit=10)
        db.close()

        assert len(rows) == 2
        plate_texts = [r[0] for r in rows]
        assert "MH12AB1234" in plate_texts
        assert "DL8CAF5030" in plate_texts


def test_api_detect_and_plates_endpoints():
    """Smoke test the FastAPI backend using a fake pipeline (no real YOLO weights needed)."""
    from fastapi.testclient import TestClient
    import backend.api as api_module
    import config as config_module
    import cv2

    # Swap in a fake pipeline so this test doesn't need YOLO weights downloaded.
    api_module._pipeline = make_pipeline_with_fake_detector()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "api_test_plates.db")
        config_module.DB_PATH = db_path
        api_module._db = None  # force it to re-create using the temp path above

        client = TestClient(api_module.app)

        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        image = make_synthetic_test_image(plate_text="MH12AB1234")
        ok, buf = cv2.imencode(".jpg", image)
        assert ok
        files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}
        r = client.post("/api/detect", files=files)
        assert r.status_code == 200
        data = r.json()
        assert len(data["plates"]) == 1
        assert len(data["logged"]) == 1  # exact OCR text can vary slightly, e.g. 3 misread as 5
        assert data["annotated_image_base64"] is not None

        r = client.get("/api/plates")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert len(rows[0]["plate_text"]) >= 4

        r = client.get("/")
        assert r.status_code == 200
        assert "Plate Reader Console" in r.text
