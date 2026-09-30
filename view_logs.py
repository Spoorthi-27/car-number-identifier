#!/usr/bin/env python3
"""
view_logs.py
------------
Print the most recently logged plates from the database.

Usage:
    python view_logs.py
    python view_logs.py --limit 50
"""
import argparse

import config
from src.database import PlateDatabase
from src.logging_setup import get_logger

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="View recent logged plates")
    parser.add_argument("--limit", type=int, default=20, help="How many recent entries to show")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        db = PlateDatabase(config.DB_PATH)
        rows = db.recent_plates(limit=args.limit)
        db.close()
    except Exception as exc:
        logger.error("Failed to read plate logs: %s", exc)
        print(f"Could not read the plate log: {exc}")
        return

    if not rows:
        print("No plates logged yet.")
        return

    for row in rows:
        confidence = row["confidence"]
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, float) else "-"
        print("-" * 32)
        print(f"Vehicle Number: {row['plate_text'] or '-'}")
        print(f"State: {row['state'] or '-'}")
        print(f"RTO Code: {row['rto_code'] or '-'}")
        print(f"District: {row['district'] or 'Unknown RTO/District'}")
        print(f"Plate Color: {row['plate_color'] or 'UNKNOWN'}")
        print(f"Detection Confidence: {confidence_text}")
        if row["ocr_confidence"] is not None:
            print(f"OCR Confidence: {row['ocr_confidence']:.2f}")
        print(f"Source: {row['source'] or '-'}")
        print(f"Date: {row['date']}")
        print(f"Time: {row['time']}")
        if row["ocr_raw"] and row["ocr_raw"] != row["plate_text"]:
            print(f"Raw OCR: {row['ocr_raw']}")
        if row["image_path"]:
            print(f"Snapshot: {row['image_path']}")
    print("-" * 32)


if __name__ == "__main__":
    main()
