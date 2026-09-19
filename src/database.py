"""
database.py
-----------
Lightweight SQLite storage for recognized plates. No server to set up -
it's just a single .db file on disk.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class PlateDatabase:
    def __init__(self, db_path: str = "database/plates.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
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

    def log_plate(
        self,
        plate_text: str,
        confidence: Optional[float] = None,
        image_path: Optional[str] = None,
    ) -> int:
        """Insert a new plate read, stamped with the current time. Returns the new row id."""
        cur = self.conn.execute(
            "INSERT INTO plate_reads (plate_text, confidence, timestamp, image_path) "
            "VALUES (?, ?, ?, ?)",
            (plate_text, confidence, datetime.now().isoformat(timespec="seconds"), image_path),
        )
        self.conn.commit()
        return cur.lastrowid

    def recent_plates(self, limit: int = 20) -> List[Tuple]:
        cur = self.conn.execute(
            "SELECT id, plate_text, confidence, timestamp, image_path "
            "FROM plate_reads ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def delete_plate(self, plate_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM plate_reads WHERE id = ?", (plate_id,))
        self.conn.commit()
        return cur.rowcount > 0
    def close(self):
        self.conn.close()