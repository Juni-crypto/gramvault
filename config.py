"""Central configuration — loads .env and provides typed access."""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_log = logging.getLogger("instaintel.config")


def _parse_allowed_users() -> list[int]:
    """Parse TELEGRAM_ALLOWED_USERS with validation."""
    raw = os.getenv("TELEGRAM_ALLOWED_USERS", "")
    if not raw.strip():
        return []
    users = []
    for uid in raw.split(","):
        uid = uid.strip()
        if uid.isdigit():
            users.append(int(uid))
        elif uid:
            _log.warning("Invalid TELEGRAM_ALLOWED_USERS entry: '%s' (skipped)", uid)
    return users


class Config:
    # Vision (images)
    VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "auto")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Gemini (video / reels)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_VIDEO_MODEL: str = os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-flash")
    GEMINI_INLINE_MAX_MB: int = int(os.getenv("GEMINI_INLINE_MAX_MB", "20"))
    GEMINI_UPLOAD_TIMEOUT: int = int(os.getenv("GEMINI_UPLOAD_TIMEOUT", "120"))

    # Anthropic model for RAG chat
    ANTHROPIC_CHAT_MODEL: str = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_ALLOWED_USERS: list[int] = _parse_allowed_users()

    # Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    MEDIA_DIR: Path = Path(os.getenv("MEDIA_DIR", "./data/media"))
    DB_PATH: Path = DATA_DIR / "instaintel.db"
    CHROMA_DIR: Path = DATA_DIR / "chroma"
    GRAPH_PATH: Path = DATA_DIR / "knowledge_graph.json"

    # Processing
    REEL_KEYFRAME_INTERVAL: int = max(1, int(os.getenv("REEL_KEYFRAME_INTERVAL", "3") or "3"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Search scoring weights
    SEARCH_GRAPH_WEIGHT: float = float(os.getenv("SEARCH_GRAPH_WEIGHT", "0.3"))
    SEARCH_TEXT_WEIGHT: float = float(os.getenv("SEARCH_TEXT_WEIGHT", "0.2"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.MEDIA_DIR, cls.CHROMA_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        errors = []
        warnings = []

        # Required
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is required")

        # Recommended
        if not cls.ANTHROPIC_API_KEY:
            warnings.append("ANTHROPIC_API_KEY not set -- entity extraction will use regex fallback")
        if not cls.GEMINI_API_KEY:
            warnings.append("GEMINI_API_KEY not set -- reels will use keyframe fallback")

        # Security
        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_ALLOWED_USERS:
            warnings.append("TELEGRAM_ALLOWED_USERS is empty -- bot will deny all users by default")

        for w in warnings:
            _log.warning("Config: %s", w)
        if errors:
            raise ValueError("Config errors:\n" + "\n".join(f"  - {e}" for e in errors))
