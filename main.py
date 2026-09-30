#!/usr/bin/env python3
"""
main.py
-------
Live pipeline in an OpenCV window:

    camera -> YOLO plate box -> crop -> color -> OCR -> district -> SQLite

Controls while running:
    q  -  quit
    s  -  manually save a snapshot of the current frame to output/

Usage:
    python main.py                  # default webcam (config.CAMERA_INDEX)
    python main.py --camera 1       # a different camera index
    python main.py --video path.mp4 # a video file instead of a live camera
"""
import argparse
import time
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config
from src.camera import CameraError, VideoOpenError, open_camera, open_video, release_capture
from src.database import PlateDatabase
from src.logging_setup import get_logger
from src.pipeline import PlateReaderPipeline
from src.records import is_recent_duplicate, mark_logged, save_plate_read

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Live Car Number Plate Reader")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index to use")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file (instead of a live camera)")
    return parser.parse_args()


def _open_source(args):
    if args.video:
        try:
            return open_video(args.video), "Video"
        except VideoOpenError as exc:
            print(exc)
            return None, None
    try:
        return open_camera(args.camera, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT), "Webcam"
    except CameraError as exc:
        print(exc)
        return None, None


def main():
    args = parse_args()
    print("Initializing pipeline (this loads the YOLO model)...")
    pipeline = PlateReaderPipeline(
        model_name=config.MODEL_NAME,
        conf_threshold=config.CONF_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        tesseract_cmd=config.TESSERACT_CMD,
        min_plate_text_length=config.MIN_PLATE_TEXT_LENGTH,
    )
    db = PlateDatabase(config.DB_PATH)
    capture, source = _open_source(args)
    if capture is None:
        db.close()
        return

    print("Running. Press 'q' in the video window to quit, 's' to save a snapshot.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("End of stream / camera read failed.")
                break

            try:
                results = pipeline.process(frame)
            except Exception as exc:
                logger.error("Frame processing failed: %s", exc)
                results = []
            annotated = pipeline.annotate(frame, results)

            for result in results:
                if is_recent_duplicate(result.text, config.DEDUPE_SECONDS):
                    continue
                outcome = save_plate_read(db, result, source, frame)
                if outcome["saved"]:
                    mark_logged(result.text)
                    print(
                        f"[LOGGED] plate='{result.text}' district='{result.district}' "
                        f"color={result.plate_color} conf={result.detection_conf:.2f}"
                    )

            cv2.imshow("Car Number Plate Reader (q = quit, s = save snapshot)", annotated)
            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            if key == ord("q"):
                break
            if key == ord("s"):
                manual_path = f"output/manual_snapshot_{int(now)}.jpg"
                Path("output").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(manual_path, annotated)
                print(f"Saved manual snapshot: {manual_path}")
    finally:
        release_capture(capture)
        cv2.destroyAllWindows()
        db.close()
        logger.info("Camera stopped")
        print("Stopped.")


if __name__ == "__main__":
    main()
