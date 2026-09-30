"""Logs stay on disk and old databases gain the plate_color column."""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import PlateDatabase


def test_logs_survive_reopening_the_database():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "plates.db")
        db = PlateDatabase(db_path)
        db.log_plate(
            "KA09AB1234",
            confidence=0.94,
            status="VALID",
            plate_color="WHITE",
            state_name="Karnataka",
            state_code="KA",
            rto_code="KA09",
            rto_name="Mysuru West",
            district="Mysuru",
            source="Webcam",
            ocr_raw="KAO9AB1234",
            ocr_confidence=0.81,
        )
        db.close()

        reopened = PlateDatabase(db_path)
        rows = reopened.recent_plates(limit=5)
        reopened.close()

        assert len(rows) == 1
        row = rows[0]
        assert row["plate_text"] == "KA09AB1234"
        assert row["district"] == "Mysuru"
        assert row["state"] == "Karnataka"
        assert row["rto_code"] == "KA09"
        assert row["plate_color"] == "WHITE"
        assert row["source"] == "Webcam"
        assert row["ocr_raw"] == "KAO9AB1234"


def test_old_database_gains_plate_color_without_losing_rows():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "legacy.db"
        connection = sqlite3.connect(db_path)
        connection.execute(
            """
            CREATE TABLE plate_reads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                confidence REAL,
                timestamp TEXT NOT NULL,
                image_path TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO plate_reads (plate_text, confidence, timestamp) VALUES (?, ?, ?)",
            ("KA11CD5678", 0.8, "2026-09-22T12:30:15"),
        )
        connection.commit()
        connection.close()

        db = PlateDatabase(str(db_path))
        rows = db.recent_plates(limit=5)
        db.close()

        assert len(rows) == 1
        assert rows[0]["plate_text"] == "KA11CD5678"
        assert rows[0]["plate_color"] == "UNKNOWN"
        assert rows[0]["district"] == "Mandya"
        assert rows[0]["rto_code"] == "KA11"
        assert rows[0]["date"] == "22-09-2026"
        assert rows[0]["time"] == "12:30:15"
