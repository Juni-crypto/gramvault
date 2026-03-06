"""Shared data models for the InstaIntel pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MediaType(Enum):
    IMAGE = "image"
    CAROUSEL = "carousel"
    REEL = "reel"
    UNKNOWN = "unknown"


@dataclass
class MediaItem:
    """A single piece of Instagram content to process."""
    media_id: str                          # Shortcode (used as primary key)
    shortcode: str                         # e.g. "B4xyz..."
    media_type: MediaType = MediaType.UNKNOWN
    url: str = ""                          # Full Instagram URL
    caption: str = ""
    author_username: str = ""
    author_id: int = 0
    timestamp: str = ""
    hashtags: list[str] = field(default_factory=list)
