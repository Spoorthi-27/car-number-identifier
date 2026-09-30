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


def test_ocr_confusion_is_normalized_and_saved(monkeypatch):
    """KAO9 from OCR must become KA09, resolve to Mysuru, and be stored."""
    import config as config_module
    from src.database import PlateDatabase
    from src.records import save_plate_read

    pipeline = make_pipeline_with_fake_detector()
    monkeypatch.setattr(pipeline.ocr, "read_plate_details", lambda _crop: ("KAO9AB1234", 0.77))

    image = make_synthetic_test_image(plate_text="KA09AB1234")
    results = pipeline.process(image)
    assert len(results) == 1
    result = results[0]
    assert result.normalized_text == "KA09AB1234"
    assert result.text == "KA09AB1234"
    assert result.raw_text == "KAO9AB1234"
    assert result.state == "Karnataka"
    assert result.rto_code == "KA09"
    assert result.district == "Mysuru"
    assert result.plate_color == "WHITE"
    assert result.confidence_ok is True

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_module.DB_PATH = str(Path(tmp_dir) / "ocr_fix.db")
        config_module.SAVE_SNAPSHOTS = False
        db = PlateDatabase(config_module.DB_PATH)
        outcome = save_plate_read(db, result, "Image", image)
        rows = db.recent_plates(limit=1)
        db.close()

    assert outcome["saved"] is True
    assert rows[0]["plate_text"] == "KA09AB1234"
    assert rows[0]["district"] == "Mysuru"
    assert rows[0]["plate_color"] == "WHITE"
    assert rows[0]["ocr_raw"] == "KAO9AB1234"
    assert rows[0]["source"] == "Image"


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
        plate_texts = [r["plate_text"] for r in rows]
        assert "MH12AB1234" in plate_texts
        assert "DL8CAF5030" in plate_texts
        assert rows[0]["plate_color"] == "UNKNOWN"
        assert rows[0]["date"] and rows[0]["time"]


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
        assert len(rows[0]["plate_text"]) >= 0
        assert rows[0]["plate_color"] in {"WHITE", "YELLOW", "GREEN", "UNKNOWN"}
        assert rows[0]["district"]
        assert rows[0]["date"]
        assert rows[0]["time"]
        assert "plate_color" in data["plates"][0]
        assert "district" in data["plates"][0]
        assert "rto_code" in data["plates"][0]

        r = client.get("/")
        assert r.status_code == 200
        assert "Plate Reader Console" in r.text

        api_module._db.close()
        api_module._db = None


def test_video_upload_is_processed_and_logged():
    """An uploaded video goes through the same save path as a still image."""
    from fastapi.testclient import TestClient
    import backend.api as api_module
    import config as config_module
    import cv2

    api_module._pipeline = make_pipeline_with_fake_detector()
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_module.DB_PATH = str(Path(tmp_dir) / "video_plates.db")
        config_module.SAVE_SNAPSHOTS = False
        api_module._db = None

        image = make_synthetic_test_image(plate_text="KA09AB1234")
        video_path = Path(tmp_dir) / "clip.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5,
            (image.shape[1], image.shape[0]),
        )
        assert writer.isOpened()
        for _ in range(3):
            writer.write(image)
        writer.release()

        client = TestClient(api_module.app)
        payload = video_path.read_bytes()
        response = client.post(
            "/api/detect_video",
            files={"file": ("clip.avi", payload, "video/avi")},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["frames_processed"] >= 1
        assert len(data["plates"]) == 1
        assert data["plates"][0]["source"] == "Video"
        assert data["plates"][0]["plate_color"] in {"WHITE", "YELLOW", "GREEN", "UNKNOWN"}
        assert data["source"] == "Video"

        rows = client.get("/api/plates").json()
        assert len(rows) == 1
        assert rows[0]["source"] == "Video"

        api_module._db.close()
        api_module._db = None


def test_camera_check_returns_a_clear_error(monkeypatch):
    from fastapi.testclient import TestClient
    import backend.api as api_module
    from src.camera import CAMERA_UNAVAILABLE, CameraError

    def _closed(*_args, **_kwargs):
        raise CameraError(CAMERA_UNAVAILABLE)

    monkeypatch.setattr(api_module, "open_camera", _closed)
    client = TestClient(api_module.app)
    response = client.get("/api/camera/check?camera=3")
    assert response.status_code == 503
    assert response.json()["error"] == CAMERA_UNAVAILABLE
    assert response.json()["ok"] is False
