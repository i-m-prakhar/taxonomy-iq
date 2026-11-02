"""Durable, bounded SQLite audit and human label feedback. No product text stored."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                  id INTEGER PRIMARY KEY, request_id TEXT UNIQUE,
                  category TEXT NOT NULL, confidence REAL NOT NULL,
                  reviewed INTEGER NOT NULL, model_version TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                  request_id TEXT PRIMARY KEY, corrected_category TEXT NOT NULL,
                  reviewer_note TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)

    def connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def record(self, request_id: str, prediction: dict):
        with self.connect() as con:
            con.execute("""INSERT OR IGNORE INTO predictions
              (request_id, category, confidence, reviewed, model_version, created_at)
              VALUES (?, ?, ?, ?, ?, ?)""",
              (request_id, prediction["category"], prediction["confidence"],
               int(prediction["needs_review"]), prediction["model_version"],
               datetime.now(timezone.utc).isoformat()))

    def feedback(self, request_id: str, label: str, note: str):
        with self.connect() as con:
            exists = con.execute("SELECT 1 FROM predictions WHERE request_id=?", (request_id,)).fetchone()
            if not exists: return False
            con.execute("""INSERT INTO feedback VALUES (?, ?, ?, ?)
                         ON CONFLICT(request_id) DO UPDATE SET corrected_category=excluded.corrected_category,
                         reviewer_note=excluded.reviewer_note, created_at=excluded.created_at""",
                        (request_id, label, note, datetime.now(timezone.utc).isoformat()))
        return True

    def distribution(self):
        with self.connect() as con:
            rows = con.execute("SELECT category, count(*) FROM predictions GROUP BY category").fetchall()
        return dict(rows)
