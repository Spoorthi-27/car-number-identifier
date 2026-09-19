import base64
import os
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from src.database import PlateDatabase
from src.district import get_district
from src.pipeline import PlateReaderPipeline

APP_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = APP_ROOT / "frontend"

app = FastAPI(title="Car Number Plate Reader API")

_pipeline: Optional[PlateReaderPipeline] = None
_db: Optional[PlateDatabase] = None


def get_pipeline() -> PlateReaderPipeline:
    """Lazily create the pipeline (loads/downloads the YOLO model on first call)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PlateReaderPipeline(
            model_name=config.MODEL_NAME,
            conf_threshold=config.CONF_THRESHOLD,
            iou_threshold=config.IOU_THRESHOLD,
            min_plate_text_length=config.MIN_PLATE_TEXT_LENGTH,
        )
    return _pipeline


def get_db() -> PlateDatabase:
    global _db
    if _db is None:
        _db = PlateDatabase(config.DB_PATH)
    return _db


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """Run the full pipeline on a single uploaded image and log any confident reads."""
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode the uploaded file as an image.")

    pipeline = get_pipeline()
    results = pipeline.process(image)
    annotated = pipeline.annotate(image, results)

    ok, buf = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None

    db = get_db()
    logged = []
    for r in results:
        if r.confidence_ok:
            db.log_plate(r.text, confidence=r.detection_conf)
            logged.append(r.text)

    return {
        "plates": [
            {
                "text": r.text,
                "box": r.box,
                "detection_conf": round(r.detection_conf, 3),
                "confidence_ok": r.confidence_ok,
                "district": get_district(r.text),
            }
            for r in results
        ],
        "logged": logged,
        "annotated_image_base64": annotated_b64,
    }


@app.get("/api/plates")
def list_plates(limit: int = 50):
    db = get_db()
    rows = db.recent_plates(limit=limit)
    return [
        {
            "id": row[0],
            "plate_text": row[1],
            "confidence": row[2],
            "timestamp": row[3],
            "image_path": row[4],
            "district": get_district(row[1]),
        }
        for row in rows
    ]


@app.delete("/api/plates/{plate_id}")
def delete_plate(plate_id: int):
    db = get_db()
    deleted = db.delete_plate(plate_id)
    if not deleted:
        raise HTTPException(404, "Plate log entry not found.")
    return {"deleted": plate_id}

def _open_camera(camera_index: int) -> cv2.VideoCapture:
    """
    Open a webcam. On Windows, OpenCV's default backend (Media Foundation)
    frequently fails to open a camera that works fine in every other app -
    DirectShow is far more reliable there, so try it first.
    """
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def _mjpeg_frames(cap: cv2.VideoCapture, camera_index: int):
    pipeline = get_pipeline()
    db = get_db()

    last_logged = {}
    if config.SAVE_SNAPSHOTS:
        Path(config.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pipeline.process(frame)
        annotated = pipeline.annotate(frame, results)

        now = time.time()
        for r in results:
            if not r.confidence_ok:
                continue
            if now - last_logged.get(r.text, 0) < config.DEDUPE_SECONDS:
                continue

            snapshot_path = None
            if config.SAVE_SNAPSHOTS:
                x, y, w, h = r.box
                crop = frame[y:y + h, x:x + w]
                snapshot_path = str(Path(config.SNAPSHOT_DIR) / f"{r.text}_{int(now)}.jpg")
                cv2.imwrite(snapshot_path, crop)

            db.log_plate(r.text, confidence=r.detection_conf, image_path=snapshot_path)
            last_logged[r.text] = now

        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        frame_bytes = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/api/video_feed")
def video_feed(camera: int = config.CAMERA_INDEX):
    # Open (and validate) the camera before starting the stream, so a bad
    # camera index or an in-use webcam returns a clear error immediately
    # instead of the frontend just hanging on a blank feed forever.
    cap = _open_camera(camera)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            503,
            f"Could not open camera index {camera}. It may be in use by "
            "another app, or this index doesn't exist - try a different "
            "number (0, 1, 2...) in the 'cam' field.",
        )

    def _stream():
        try:
            yield from _mjpeg_frames(cap, camera)
        finally:
            cap.release()

    return StreamingResponse(
        _stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/snapshot/{filename}")
def get_snapshot(filename: str):
    path = Path(config.SNAPSHOT_DIR) / filename
    if not path.exists() or path.parent.resolve() != Path(config.SNAPSHOT_DIR).resolve():
        raise HTTPException(404, "Snapshot not found.")
    return FileResponse(path)


# Serve the frontend last, so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
