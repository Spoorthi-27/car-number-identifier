#!/usr/bin/env python3
"""
main.py
-------
Live pipeline, exactly as described:

    camera -> capture frame by frame (OpenCV)
           -> YOLO detects the plate, draws a box
           -> crop out just the plate region
           -> enhance the crop (resize, denoise, brightness/contrast)
           -> OCR reads the characters
           -> text is shown on screen and logged to a local SQLite database

Controls while running:
    q  -  quit
    s  -  manually save a snapshot of the current frame to output/

Usage:
    python main.py                  # use the default webcam (config.CAMERA_INDEX)
    python main.py --camera 1       # use a different camera index
    python main.py --video path.mp4 # run on a video file instead of a live camera
"""
import argparse
import time
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config
from src.pipeline import PlateReaderPipeline
from src.database import PlateDatabase


def parse_args():
    parser = argparse.ArgumentParser(description="Live Car Number Plate Reader")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index to use")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file (instead of a live camera)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Initializing pipeline (this loads the YOLO model)...")
    pipeline = PlateReaderPipeline(
        model_name=config.MODEL_NAME,
        conf_threshold=config.CONF_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        min_plate_text_length=config.MIN_PLATE_TEXT_LENGTH,
    )
    db = PlateDatabase(config.DB_PATH)

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        return

    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if config.SAVE_SNAPSHOTS:
        Path(config.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    last_logged = {}  # plate_text -> unix timestamp it was last logged

    print("Running. Press 'q' in the video window to quit, 's' to save a snapshot.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream / camera read failed.")
            break

        frame_count += 1
        results = pipeline.process(frame)
        annotated = pipeline.annotate(frame, results)

        now = time.time()
        for r in results:
            if not r.confidence_ok:
                continue

            last_time = last_logged.get(r.text, 0)
            if now - last_time < config.DEDUPE_SECONDS:
                continue  # already logged this plate very recently, skip

            snapshot_path = None
            if config.SAVE_SNAPSHOTS:
                x, y, w, h = r.box
                crop = frame[y:y + h, x:x + w]
                snapshot_path = str(Path(config.SNAPSHOT_DIR) / f"{r.text}_{int(now)}.jpg")
                cv2.imwrite(snapshot_path, crop)

            db.log_plate(r.text, confidence=r.detection_conf, image_path=snapshot_path)
            last_logged[r.text] = now
            print(f"[LOGGED] plate='{r.text}'  conf={r.detection_conf:.2f}  snapshot={snapshot_path}")

        cv2.imshow("Car Number Plate Reader (q = quit, s = save snapshot)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            manual_path = f"output/manual_snapshot_{int(now)}.jpg"
            cv2.imwrite(manual_path, annotated)
            print(f"Saved manual snapshot: {manual_path}")

    cap.release()
    cv2.destroyAllWindows()
    db.close()
    print("Stopped.")


if __name__ == "__main__":
    main()
