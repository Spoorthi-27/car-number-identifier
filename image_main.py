#!/usr/bin/env python3
"""
image_main.py
-------------
Run the pipeline on a single saved image instead of a live camera.
Useful for quick tests without needing a webcam handy.

Usage:
    python image_main.py --image sample_images/car.jpg
    python image_main.py --image sample_images/car.jpg --output output/result.jpg
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config
from src.pipeline import PlateReaderPipeline
from src.database import PlateDatabase
from src.utils import load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Car Number Plate Reader (single image)")
    parser.add_argument("--image", type=str, required=True, help="Path to a car image")
    parser.add_argument("--output", type=str, default="output/result.jpg", help="Where to save the annotated image")
    parser.add_argument("--no-log", action="store_true", help="Don't write results to the database")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = PlateReaderPipeline(
        model_name=config.MODEL_NAME,
        conf_threshold=config.CONF_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        min_plate_text_length=config.MIN_PLATE_TEXT_LENGTH,
    )
    image = load_image(args.image)
    results = pipeline.process(image)

    if not results:
        print("No plates detected.")
    else:
        db = None if args.no_log else PlateDatabase(config.DB_PATH)
        for i, r in enumerate(results, start=1):
            status = "OK" if r.confidence_ok else "low-confidence"
            print(f"  [{i}] box={r.box}  text='{r.text}'  det_conf={r.detection_conf:.2f}  ({status})")
            if db and r.confidence_ok:
                db.log_plate(r.text, confidence=r.detection_conf)
        if db:
            db.close()

    annotated = pipeline.annotate(image, results)
    save_image(args.output, annotated)
    print(f"Annotated image saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
