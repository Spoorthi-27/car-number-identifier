"""
database.py
-----------
SQLite storage for recognized plates. The file is created on first use and
kept across restarts. New columns are added with ALTER TABLE so an existing
plates.db is not rebuilt or wiped.
"""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from .vehicle_type import UNKNOWN_VEHICLE_TYPE, classify_vehicle_type
from .district import lookup_registration
from .logging_setup import get_logger

logger = get_logger()


def split_timestamp(timestamp: Optional[str]) -> tuple:
    """ISO timestamp -> ('DD-MM-YYYY', 'HH:MM:SS')."""
    if not timestamp:
        return "-", "-"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp, "-"
    return parsed.strftime("%d-%m-%Y"), parsed.strftime("%H:%M:%S")


class PlateDatabase:
    def __init__(self, db_path: str = "database/plates.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_table()

    def _create_table(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_reads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    confidence REAL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT
                )
                """
            )
            self.conn.commit()
            self._ensure_column("status", "status TEXT NOT NULL DEFAULT 'VALID'")
            self._ensure_column("plate_color", "plate_color TEXT NOT NULL DEFAULT 'UNKNOWN'")
            self._ensure_column("state_name", "state_name TEXT")
            self._ensure_column("state_code", "state_code TEXT")
            self._ensure_column("rto_code", "rto_code TEXT")
            self._ensure_column("rto_name", "rto_name TEXT")
            self._ensure_column("district", "district TEXT")
            self._ensure_column("ocr_confidence", "ocr_confidence REAL")
            self._ensure_column("ocr_raw", "ocr_raw TEXT")
            self._ensure_column("source", "source TEXT")
            self._ensure_column("vehicle_type", f"vehicle_type TEXT NOT NULL DEFAULT '{UNKNOWN_VEHICLE_TYPE}'")
            self._backfill_registration()
            self._backfill_vehicle_type()
    def _ensure_column(self, name: str, definition: str) -> None:
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(plate_reads)")}
        if name in existing:
            return
        self.conn.execute(f"ALTER TABLE plate_reads ADD COLUMN {definition}")
        self.conn.commit()
        logger.info("Added database column %s", name)

    def _backfill_registration(self) -> None:
        """Fill district/state on rows saved before those columns existed."""
        rows = self.conn.execute(
            """
            SELECT id, plate_text FROM plate_reads
            WHERE district IS NULL OR district = ''
            """
        ).fetchall()
        for row_id, plate_text in rows:
            info = lookup_registration(plate_text)
            self.conn.execute(
                """
                UPDATE plate_reads
                SET state_name = ?, state_code = ?, rto_code = ?, rto_name = ?, district = ?
                WHERE id = ?
                """,
                (info.state, info.state_code, info.rto_code, info.rto_name, info.district, row_id),
            )
        if rows:
            self.conn.commit()
            logger.info("Backfilled district details for %s existing log row(s)", len(rows))
    def _backfill_vehicle_type(self) -> None:
        """Derive vehicle_type from the stored plate_color on rows saved before it existed."""
        rows = self.conn.execute(
            """
            SELECT id, plate_color FROM plate_reads
            WHERE vehicle_type IS NULL OR vehicle_type = '' OR vehicle_type = ?
            """,
            (UNKNOWN_VEHICLE_TYPE,),
        ).fetchall()
        changed = 0
        for row_id, plate_color in rows:
            vehicle_type = classify_vehicle_type(plate_color)
            if vehicle_type == UNKNOWN_VEHICLE_TYPE:
                continue
            self.conn.execute("UPDATE plate_reads SET vehicle_type = ? WHERE id = ?", (vehicle_type, row_id))
            changed += 1
        if changed:
            self.conn.commit()
            logger.info("Backfilled vehicle type for %s existing log row(s)", changed)
    def log_plate(
        self,
        plate_text: str,
        confidence: Optional[float] = None,
        image_path: Optional[str] = None,
        status: str = "VALID",
        plate_color: str = "UNKNOWN",
        state_name: Optional[str] = None,
        state_code: Optional[str] = None,
        rto_code: Optional[str] = None,
        rto_name: Optional[str] = None,
        district: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
        ocr_raw: Optional[str] = None,
        source: Optional[str] = None,
        vehicle_type: Optional[str] = None,
    ) -> int:
        """Insert one plate read and commit it. Returns the new row id."""
        color = (plate_color or "UNKNOWN").upper()
        if color not in {"WHITE", "YELLOW", "GREEN", "UNKNOWN"}:
            color = "UNKNOWN"
        # Callers may pass vehicle_type; otherwise it is derived from the color.
        vehicle_type = vehicle_type or classify_vehicle_type(color)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            try:
                cursor = self.conn.execute(
                    """
                    INSERT INTO plate_reads (
                        plate_text, confidence, timestamp, image_path, status,
                        plate_color, state_name, state_code, rto_code, rto_name,
                        district, ocr_confidence, ocr_raw, source, vehicle_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plate_text or "",
                        confidence,
                        timestamp,
                        image_path,
                        status or "OCR_UNCERTAIN",
                        color,
                        state_name,
                        state_code,
                        rto_code,
                        rto_name,
                        district,
                        ocr_confidence,
                        ocr_raw,
                        source,
                        vehicle_type,
                    ),
                )
                self.conn.commit()
                return int(cursor.lastrowid)
            except Exception:
                self.conn.rollback()
                raise

    def recent_plates(self, limit: int = 20) -> List[dict]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, plate_text, confidence, timestamp, image_path, status,
                       plate_color, state_name, state_code, rto_code, rto_name,
                       district, ocr_confidence, ocr_raw, source, vehicle_type
                FROM plate_reads
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        results = []
        for row in rows:
            date_text, time_text = split_timestamp(row[3])
            results.append(
                {
                    "id": row[0],
                    "plate_text": row[1],
                    "confidence": row[2],
                    "timestamp": row[3],
                    "image_path": row[4],
                    "status": row[5],
                    "plate_color": row[6] or "UNKNOWN",
                    "state": row[7],
                    "state_code": row[8],
                    "rto_code": row[9],
                    "rto_name": row[10],
                    "district": row[11] or "Unknown RTO/District",
                    "ocr_confidence": row[12],
                    "ocr_raw": row[13],
                    "source": row[14],
                    "vehicle_type": row[15] or classify_vehicle_type(row[6]),
                    "date": date_text,
                    "time": time_text,
                }
            )
        return results

    def delete_plate(self, plate_id: int) -> bool:
        with self._lock:
            cursor = self.conn.execute("DELETE FROM plate_reads WHERE id = ?", (plate_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self.conn.close()
    _ROW_COLUMNS = (
        "id, plate_text, confidence, timestamp, image_path, status, "
        "plate_color, state_name, state_code, rto_code, rto_name, "
        "district, ocr_confidence, ocr_raw, source, vehicle_type"
    )

    @staticmethod
    def _row_to_dict(row) -> dict:
        """Shared by recent_plates and get_plate so the mapping exists in one place."""
        date_text, time_text = split_timestamp(row[3])
        return {
            "id": row[0],
            "plate_text": row[1],
            "confidence": row[2],
            "timestamp": row[3],
            "image_path": row[4],
            "status": row[5],
            "plate_color": row[6] or "UNKNOWN",
            "state": row[7],
            "state_code": row[8],
            "rto_code": row[9],
            "rto_name": row[10],
            "district": row[11] or "Unknown RTO/District",
            "ocr_confidence": row[12],
            "ocr_raw": row[13],
            "source": row[14],
            "vehicle_type": row[15] or classify_vehicle_type(row[6]),
            "date": date_text,
            "time": time_text,
        }

    def recent_plates(self, limit: int = 20) -> List[dict]:
        with self._lock:
            cursor = self.conn.execute(
                f"""
                SELECT {self._ROW_COLUMNS}
                FROM plate_reads
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_plate(self, plate_id: int) -> Optional[dict]:
        """A single logged read by id, in the same shape as recent_plates(). None if not found."""
        with self._lock:
            cursor = self.conn.execute(
                f"""
                SELECT {self._ROW_COLUMNS}
                FROM plate_reads
                WHERE id = ?
                """,
                (plate_id,),
            )
            row = cursor.fetchone()
        return self._row_to_dict(row) if row else None