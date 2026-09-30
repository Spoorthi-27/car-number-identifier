#!/usr/bin/env python3
"""
image_main.py
-------------
Run the pipeline on a single saved image instead of a live camera.

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
from src.database import PlateDatabase
from src.logging_setup import get_logger
from src.pipeline import PlateReaderPipeline
from src.records import save_plate_read
from src.utils import load_image, save_image

logger = get_logger()


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
        tesseract_cmd=config.TESSERACT_CMD,
        min_plate_text_length=config.MIN_PLATE_TEXT_LENGTH,
    )
    try:
        image = load_image(args.image)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(exc)
        return

    results = pipeline.process(image)
    if not results:
        print("No plates detected.")
    else:
        db = None if args.no_log else PlateDatabase(config.DB_PATH)
        for index, result in enumerate(results, start=1):
            print(
                f"  [{index}] text='{result.text}' raw='{result.raw_text}' "
                f"district='{result.district}' color={result.plate_color} "
                f"det_conf={result.detection_conf:.2f} status={result.status}"
            )
            if db is not None:
                outcome = save_plate_read(db, result, "Image", image)
                if not outcome["saved"]:
                    print(f"      save failed: {outcome['error']}")
        if db is not None:
            db.close()

    annotated = pipeline.annotate(image, results)
    save_image(args.output, annotated)
    print(f"Annotated image saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
