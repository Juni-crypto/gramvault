"""SQLite Database — Stores metadata, processing state, and content records.

Tables:
- posts: Core post metadata
- slides: Individual carousel slides / reel keyframes
- processed_messages: DM messages already handled (dedup)
- entities: Extracted entities linked to posts
"""

import sqlite3
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from config import Config
from utils.logger import log


class Database:
    """SQLite storage layer for InstaIntel."""

    def __init__(self):
        Config.ensure_dirs()
        self.db_path = Config.DB_PATH
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        log.info("Database initialized: %s", self.db_path)

    def _create_tables(self):
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                media_id      TEXT PRIMARY KEY,
                shortcode     TEXT UNIQUE,
                media_type    TEXT,
                url           TEXT,
                caption       TEXT DEFAULT '',
                author_username TEXT DEFAULT '',
                author_id     INTEGER DEFAULT 0,
                post_timestamp TEXT DEFAULT '',
                hashtags      TEXT DEFAULT '[]',
                ocr_text      TEXT DEFAULT '',
                description   TEXT DEFAULT '',
                category      TEXT DEFAULT 'general',
                summary       TEXT DEFAULT '',
                entities_json TEXT DEFAULT '{}',
                dm_message_id TEXT DEFAULT '',
                processed_at  TEXT DEFAULT '',
                status        TEXT DEFAULT 'pending'
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS slides (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id      TEXT REFERENCES posts(media_id),
                slide_index   INTEGER,
                image_path    TEXT,
                ocr_text      TEXT DEFAULT '',
                description   TEXT DEFAULT '',
                provider      TEXT DEFAULT ''
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_id    TEXT PRIMARY KEY,
                processed_at  TEXT DEFAULT ''
            )
        """)

        self.conn.commit()

    # ─── Posts ────────────────────────────────────────────────────────

    def upsert_post(self, **kwargs) -> None:
        """Insert or update a post record."""
        # Serialize hashtags list
        if "hashtags" in kwargs and isinstance(kwargs["hashtags"], list):
            kwargs["hashtags"] = json.dumps(kwargs["hashtags"])
        if "entities_json" in kwargs and isinstance(kwargs["entities_json"], dict):
            kwargs["entities_json"] = json.dumps(kwargs["entities_json"])

        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        updates = ", ".join(f"{k}=excluded.{k}" for k in kwargs.keys() if k != "media_id")

        sql = f"""
            INSERT INTO posts ({columns}) VALUES ({placeholders})
            ON CONFLICT(media_id) DO UPDATE SET {updates}
        """
        with self._lock:
            self.conn.execute(sql, list(kwargs.values()))
            self.conn.commit()

    def get_post(self, media_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM posts WHERE media_id = ?", (media_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_posts(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM posts WHERE status = 'done' ORDER BY processed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_posts_by_category(self, category: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM posts WHERE category = ? AND status = 'done'", (category,)
        ).fetchall()
        return [dict(r) for r in rows]

    def search_posts(self, query: str) -> list[dict]:
        """Full-text search across caption, ocr_text, description, summary."""
        like = f"%{query}%"
        rows = self.conn.execute(
            """SELECT * FROM posts WHERE status = 'done' AND (
                caption LIKE ? OR ocr_text LIKE ? OR description LIKE ? OR summary LIKE ?
            ) ORDER BY processed_at DESC""",
            (like, like, like, like),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        cur = self.conn.cursor()
        total = cur.execute("SELECT COUNT(*) FROM posts WHERE status='done'").fetchone()[0]
        by_type = cur.execute(
            "SELECT media_type, COUNT(*) FROM posts WHERE status='done' GROUP BY media_type"
        ).fetchall()
        by_cat = cur.execute(
            "SELECT category, COUNT(*) FROM posts WHERE status='done' GROUP BY category ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {
            "total_posts": total,
            "by_type": {r[0]: r[1] for r in by_type},
            "by_category": {r[0]: r[1] for r in by_cat},
        }

    # ─── Slides ───────────────────────────────────────────────────────

    def add_slide(self, media_id: str, slide_index: int, image_path: str,
                  ocr_text: str = "", description: str = "", provider: str = ""):
        with self._lock:
            self.conn.execute(
                """INSERT INTO slides (media_id, slide_index, image_path, ocr_text, description, provider)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (media_id, slide_index, image_path, ocr_text, description, provider),
            )
            self.conn.commit()

    def get_slides(self, media_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM slides WHERE media_id = ? ORDER BY slide_index", (media_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── URL Dedup ─────────────────────────────────────────────────

    def is_url_processed(self, shortcode: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed_messages WHERE message_id = ?", (shortcode,)
        ).fetchone()
        return row is not None

    def mark_url_processed(self, shortcode: str):
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO processed_messages (message_id, processed_at) VALUES (?, ?)",
                (shortcode, datetime.now(timezone.utc).isoformat()),
            )
            self.conn.commit()

    # ─── Message Dedup (legacy) ──────────────────────────────────

    def is_message_processed(self, message_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed_messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        return row is not None

    def mark_message_processed(self, message_id: str):
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO processed_messages (message_id, processed_at) VALUES (?, ?)",
                (message_id, datetime.now(timezone.utc).isoformat()),
            )
            self.conn.commit()

    def get_today_posts(self) -> list[dict]:
        """Get posts processed today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self.conn.execute(
            "SELECT * FROM posts WHERE status = 'done' AND processed_at LIKE ? ORDER BY processed_at DESC",
            (f"{today}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def flush_all(self):
        """Delete all data from all tables."""
        with self._lock:
            self.conn.execute("DELETE FROM slides")
            self.conn.execute("DELETE FROM processed_messages")
            self.conn.execute("DELETE FROM posts")
            self.conn.commit()
        log.info("Database flushed — all data deleted")

    def close(self):
        self.conn.close()
