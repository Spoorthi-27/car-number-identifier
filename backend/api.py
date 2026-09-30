import base64
import csv
import io
import tempfile
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from src.camera import CAMERA_UNAVAILABLE, CameraError, VideoOpenError, open_camera, open_video, release_capture
from src.database import PlateDatabase, split_timestamp
from src.logging_setup import get_logger
from src.pipeline import PlateReaderPipeline
from src.records import is_recent_duplicate, mark_logged, save_plate_read

logger = get_logger()

APP_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = APP_ROOT / "frontend"

app = FastAPI(title="Car Number Plate Reader API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[PlateReaderPipeline] = None
_pipeline_lock = threading.Lock()
_db: Optional[PlateDatabase] = None
_db_lock = threading.Lock()
_stream_lock = threading.Lock()
_stream_stop = threading.Event()

_SOURCE_NAMES = {
    "image": "Image",
    "video": "Video",
    "webcam": "Webcam",
}


def normalize_source(value: Optional[str]) -> str:
    if not value:
        return "Image"
    return _SOURCE_NAMES.get(value.strip().lower(), "Image")


def get_pipeline() -> PlateReaderPipeline:
    """Lazily create the pipeline (loads/downloads the YOLO model on first call)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            logger.info("Loading plate detection model")
            _pipeline = PlateReaderPipeline(
                model_name=config.MODEL_NAME,
                conf_threshold=config.CONF_THRESHOLD,
                iou_threshold=config.IOU_THRESHOLD,
                tesseract_cmd=config.TESSERACT_CMD,
                min_plate_text_length=config.MIN_PLATE_TEXT_LENGTH,
            )
    return _pipeline


def get_db() -> PlateDatabase:
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is None:
            _db = PlateDatabase(config.DB_PATH)
    return _db


def _rotate_stream_stop() -> threading.Event:
    """Stop any stream already running, then return a fresh stop event."""
    global _stream_stop
    with _stream_lock:
        _stream_stop.set()
        _stream_stop = threading.Event()
        return _stream_stop


def _plate_payload(result, source: str, save_info: dict) -> dict:
    date_text, time_text = split_timestamp(datetime.now().isoformat(timespec="seconds"))
    image_path = save_info.get("image_path")
    snapshot_url = None
    if image_path:
        snapshot_url = f"/api/snapshot/{Path(image_path).name}"
    detection_conf = round(float(result.detection_conf), 3)
    return {
        "id": save_info.get("id"),
        "text": result.text,
        "plate_text": result.text,
        "raw_text": result.raw_text,
        "box": result.box,
        "detection_conf": detection_conf,
        "confidence": detection_conf,
        "ocr_confidence": None if result.ocr_confidence is None else round(float(result.ocr_confidence), 3),
        "confidence_ok": result.confidence_ok,
        "status": result.status,
        "state": result.state,
        "state_code": result.state_code,
        "rto_code": result.rto_code,
        "rto_name": result.rto_name,
        "district": result.district,
        "plate_color": result.plate_color,
        "color_confidence": result.color_confidence,
        "vehicle_type": result.vehicle_type,
        "source": source,
        "date": date_text,
        "time": time_text,
        "saved": bool(save_info.get("saved")),
        "duplicate": bool(save_info.get("duplicate")),
        "save_error": save_info.get("error"),
        "snapshot_url": snapshot_url,
        "validation_note": result.validation_note,
    }


def _store_results(results, source: str, frame, dedupe: bool) -> tuple:
    payloads = []
    logged = []
    db = get_db()
    for result in results:
        duplicate = dedupe and is_recent_duplicate(result.text, config.DEDUPE_SECONDS)
        if duplicate:
            save_info = {"saved": True, "id": None, "error": None, "image_path": None, "duplicate": True}
        else:
            save_info = save_plate_read(db, result, source, frame)
            save_info["duplicate"] = False
            if save_info["saved"]:
                if dedupe:
                    mark_logged(result.text)
                logged.append(result.text or "")
        payloads.append(_plate_payload(result, source, save_info))
    return payloads, logged


def _decode_image(contents: bytes):
    if not contents:
        raise HTTPException(400, "The uploaded file is empty.")
    array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode the uploaded file as an image.")
    return image


def _annotated_base64(image, results) -> Optional[str]:
    try:
        annotated = get_pipeline().annotate(image, results)
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as exc:
        logger.error("Failed to encode annotated image: %s", exc)
        return None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), source: str = "Image"):
    """Run the full pipeline on one uploaded image and log every detection."""
    source_name = normalize_source(source)
    try:
        contents = await file.read()
        image = _decode_image(contents)
        pipeline = get_pipeline()
        results = pipeline.process(image)
        payloads, logged = _store_results(results, source_name, image, dedupe=(source_name == "Webcam"))
        return {
            "plates": payloads,
            "logged": logged,
            "annotated_image_base64": _annotated_base64(image, results),
            "source": source_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("API error on /api/detect: %s", exc)
        raise HTTPException(500, "Plate detection failed. See server logs for details.") from exc


@app.post("/api/detect_video")
async def detect_video(file: UploadFile = File(...)):
    """Sample an uploaded video, detect plates, and log each distinct one."""
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}:
        suffix = ".mp4"
    temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temporary_path = temporary.name
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "The uploaded file is empty.")
        temporary.write(contents)
        temporary.close()
        try:
            captured, frames_processed, preview_frame, preview_results = _analyze_video(temporary_path)
        except VideoOpenError:
            raise HTTPException(400, "Video cannot be opened.") from None

        payloads = []
        logged = []
        for result, frame in captured:
            batch, batch_logged = _store_results([result], "Video", frame, dedupe=False)
            payloads.extend(batch)
            logged.extend(batch_logged)
        annotated = None
        if preview_frame is not None:
            annotated = _annotated_base64(preview_frame, preview_results)
        return {
            "plates": payloads,
            "logged": logged,
            "annotated_image_base64": annotated,
            "frames_processed": frames_processed,
            "source": "Video",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("API error on /api/detect_video: %s", exc)
        raise HTTPException(500, "Video processing failed. See server logs for details.") from exc
    finally:
        Path(temporary_path).unlink(missing_ok=True)


def _analyze_video(path: str):
    """Sample a video and keep the highest-confidence read of each plate."""
    pipeline = get_pipeline()
    capture = open_video(path)
    kept = {}
    frame_index = 0
    processed = 0
    preview_frame = None
    preview_results = []
    sample_every = max(1, config.VIDEO_SAMPLE_EVERY)
    try:
        while processed < config.VIDEO_MAX_FRAMES:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            if (frame_index - 1) % sample_every != 0:
                continue
            processed += 1
            try:
                results = pipeline.process(frame)
            except Exception as exc:
                logger.error("Failed to process video frame %s: %s", frame_index, exc)
                continue
            if results:
                preview_frame = frame
                preview_results = results
            for result in results:
                key = result.normalized_text or result.raw_text or f"unread-{frame_index}-{result.box}"
                previous = kept.get(key)
                if previous is None or result.detection_conf >= previous[0].detection_conf:
                    kept[key] = (result, frame.copy())
    finally:
        release_capture(capture)
        logger.info("Video processing finished (%s sampled frame(s))", processed)
    return list(kept.values()), processed, preview_frame, preview_results


def _row_payload(row: dict) -> dict:
    image_path = row.get("image_path")
    snapshot_url = f"/api/snapshot/{Path(image_path).name}" if image_path else None
    confidence = row.get("confidence")
    return {
        "id": row["id"],
        "plate_text": row["plate_text"],
        "text": row["plate_text"],
        "raw_text": row.get("ocr_raw"),
        "detection_conf": confidence,
        "confidence": confidence,
        "ocr_confidence": row.get("ocr_confidence"),
        "timestamp": row.get("timestamp"),
        "date": row.get("date"),
        "time": row.get("time"),
        "image_path": image_path,
        "snapshot_url": snapshot_url,
        "status": row.get("status"),
        "district": row.get("district") or "Unknown RTO/District",
        "state": row.get("state"),
        "state_code": row.get("state_code"),
        "rto_code": row.get("rto_code"),
        "rto_name": row.get("rto_name"),
        "plate_color": row.get("plate_color") or "UNKNOWN",
        "vehicle_type": row.get("vehicle_type") or "Unknown",
        "source": row.get("source"),
    }


@app.get("/api/plates")
def list_plates(limit: int = Query(default=50, ge=1, le=1000)):
    try:
        rows = get_db().recent_plates(limit=limit)
    except Exception as exc:
        logger.error("Failed to read plate logs: %s", exc)
        raise HTTPException(500, "Could not read the plate log.") from exc
    return [_row_payload(row) for row in rows]

# ---------------------------------------------------------------------
# CSV download
#
# One column mapping is shared by both the single-record and the
# history download endpoints, so the column list and formatting exist
# in exactly one place. Both endpoints read through the same
# PlateDatabase methods used by /api/plates - no database logic is
# duplicated here.
# ---------------------------------------------------------------------
def _pct(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return ""


_CSV_COLUMNS = (
    ("Vehicle Number", lambda p: p.get("plate_text") or p.get("text") or ""),
    ("State", lambda p: p.get("state") or ""),
    ("RTO Code", lambda p: p.get("rto_code") or ""),
    ("RTO Office", lambda p: p.get("rto_name") or ""),
    ("District", lambda p: p.get("district") or "Unknown RTO/District"),
    ("Plate Color", lambda p: p.get("plate_color") or "UNKNOWN"),
    ("Vehicle Type", lambda p: p.get("vehicle_type") or "Unknown"),
    ("Plate Detection Confidence", lambda p: _pct(p.get("confidence"))),
    ("OCR Confidence", lambda p: _pct(p.get("ocr_confidence"))),
    ("Raw OCR", lambda p: p.get("raw_text") or ""),
    ("Source", lambda p: p.get("source") or ""),
    ("Date", lambda p: p.get("date") or ""),
    ("Time", lambda p: p.get("time") or ""),
    ("Status", lambda p: p.get("status") or ""),
)
def _csv_safe(value) -> str:
    """Stop Excel from running text read off an image as a formula."""
    text = "" if value is None else str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text

def _plates_csv(payloads: list) -> str:
    """Render a list of row/plate payload dicts as CSV text (UTF-8 BOM, for Excel)."""
    buffer = io.StringIO()
    buffer.write("\ufeff")
    writer = csv.writer(buffer)
    writer.writerow([label for label, _ in _CSV_COLUMNS])
    for payload in payloads:
        writer.writerow([_csv_safe(getter(payload)) for _, getter in _CSV_COLUMNS])
    return buffer.getvalue()


def _csv_response(csv_text: str, filename: str) -> Response:
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _safe_filename_part(text: str, fallback: str) -> str:
    cleaned = "".join(ch for ch in (text or "") if ch.isascii() and ch.isalnum())
    return cleaned or fallback


@app.get("/api/plates/download")
def download_plate_history(limit: int = Query(default=200, ge=1, le=1000)):
    """CSV of the most recent logged reads (the Recent Reads table)."""
    try:
        rows = get_db().recent_plates(limit=limit)
    except Exception as exc:
        logger.error("Failed to read plate logs for CSV download: %s", exc)
        raise HTTPException(500, "Could not read the plate log.") from exc
    csv_text = _plates_csv([_row_payload(row) for row in rows])
    return _csv_response(csv_text, "plate_history.csv")


@app.get("/api/plates/{plate_id}/download")
def download_plate(plate_id: int):
    """CSV for one previously logged detection."""
    try:
        row = get_db().get_plate(plate_id)
    except Exception as exc:
        logger.error("Failed to read plate %s for CSV download: %s", plate_id, exc)
        raise HTTPException(500, "Could not read that plate log entry.") from exc
    if row is None:
        raise HTTPException(404, "Plate log entry not found.")
    payload = _row_payload(row)
    csv_text = _plates_csv([payload])
    plate_part = _safe_filename_part(payload.get("plate_text"), "vehicle")
    filename = f"{plate_part}_{plate_id}.csv"
    return _csv_response(csv_text, filename)


@app.delete("/api/plates/{plate_id}")
def delete_plate(plate_id: int):
    try:
        deleted = get_db().delete_plate(plate_id)
    except Exception as exc:
        logger.error("Failed to delete plate %s: %s", plate_id, exc)
        raise HTTPException(500, "Could not delete the log entry.") from exc
    if not deleted:
        raise HTTPException(404, "Plate log entry not found.")
    return {"deleted": plate_id}


@app.get("/api/camera/check")
def camera_check(camera: int = Query(default=config.CAMERA_INDEX, ge=0, le=10)):
    """Open the camera long enough to know whether frames are available."""
    try:
        capture = open_camera(
            camera,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            attempts=2,
            delay=0.15,
        )
    except CameraError:
        return JSONResponse(status_code=503, content={"ok": False, "error": CAMERA_UNAVAILABLE})
    release_capture(capture)
    logger.info("Camera check succeeded (index %s)", camera)
    return {"ok": True, "camera": camera}


@app.post("/api/camera/stop")
def stop_camera():
    _stream_stop.set()
    logger.info("Camera stop requested")
    return {"stopped": True}


def _mjpeg_frames(cap: cv2.VideoCapture, pipeline: PlateReaderPipeline, stop_event: threading.Event):
    frame_count = 0
    last_annotated = None
    while not stop_event.is_set():
        try:
            ok, frame = cap.read()
        except Exception as exc:
            logger.error("Camera error: %s", exc)
            break
        if not ok or frame is None:
            logger.error("Camera error: the camera stopped producing frames")
            break

        frame_count += 1
        run_detection = (frame_count % config.PROCESS_EVERY_N_FRAMES) == 1 or last_annotated is None
        if run_detection:
            try:
                results = pipeline.process(frame)
                _store_results(results, "Webcam", frame, dedupe=True)
                annotated = pipeline.annotate(frame, results)
                last_annotated = annotated
            except Exception as exc:
                logger.error("Frame processing failed: %s", exc)
                annotated = last_annotated if last_annotated is not None else frame
        else:
            annotated = last_annotated

        encoded, buffer = cv2.imencode(".jpg", annotated)
        if not encoded:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/api/video_feed")
def video_feed(camera: int = Query(default=config.CAMERA_INDEX, ge=0, le=10)):
    stop_event = _rotate_stream_stop()
    try:
        capture = open_camera(camera, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    except CameraError:
        raise HTTPException(503, CAMERA_UNAVAILABLE) from None

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        release_capture(capture)
        logger.error("Failed to start detection pipeline: %s", exc)
        raise HTTPException(500, "Plate detection model failed to load. See server logs.") from exc

    def _stream():
        try:
            yield from _mjpeg_frames(capture, pipeline, stop_event)
        finally:
            release_capture(capture)
            logger.info("Camera stopped")

    return StreamingResponse(_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot/{filename}")
def get_snapshot(filename: str):
    path = (Path(config.SNAPSHOT_DIR) / filename).resolve()
    root = Path(config.SNAPSHOT_DIR).resolve()
    if path.parent != root or not path.exists():
        raise HTTPException(404, "Snapshot not found.")
    return FileResponse(path)

@app.websocket("/ws/{path:path}")
async def _reject_unknown_websocket(websocket):
    # This app has no real WebSocket endpoints; something outside this
    # project (a browser extension or similar) probes /ws/... on this
    # port. Closing it cleanly avoids a noisy traceback in the console.
    await websocket.close(code=1000)
# Serve the frontend last, so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
