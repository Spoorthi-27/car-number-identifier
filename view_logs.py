#!/usr/bin/env python3
"""
view_logs.py
------------
Quickly print the most recently logged plates from the database.

Usage:
    python view_logs.py
    python view_logs.py --limit 50
"""
import argparse

import config
from src.database import PlateDatabase


def parse_args():
    parser = argparse.ArgumentParser(description="View recent logged plates")
    parser.add_argument("--limit", type=int, default=20, help="How many recent entries to show")
    return parser.parse_args()


def main():
    args = parse_args()
    db = PlateDatabase(config.DB_PATH)
    rows = db.recent_plates(limit=args.limit)
    db.close()

    if not rows:
        print("No plates logged yet.")
        return

    print(f"{'Plate':<15} {'Conf':<6} {'Timestamp':<20} Snapshot")
    print("-" * 70)
    for plate_text, confidence, timestamp, image_path in rows:
        conf_str = f"{confidence:.2f}" if confidence is not None else "-"
        print(f"{plate_text:<15} {conf_str:<6} {timestamp:<20} {image_path or '-'}")


if __name__ == "__main__":
    main()
