"""Content Downloader — Downloads Instagram content.

Two-level fallback chain (all use Proxifly free proxies):
1. instaloader + proxy (3 proxy attempts)
2. yt-dlp + proxy

Handles:
- Single image posts → downloads image
- Carousels → downloads all slides
- Reels → downloads video + extracts keyframes via ffmpeg
"""

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import instaloader
import requests

from config import Config
from core.models import MediaItem, MediaType
from core.proxy_rotator import ProxyRotator
from utils.logger import log


@dataclass
class DownloadedContent:
    """Result of downloading a media item."""
    media_item: MediaItem
    image_paths: list[Path] = field(default_factory=list)
    video_path: Path | None = None
    caption: str = ""
    success: bool = False


_SHORTCODE_RE = re.compile(r"instagram\.com/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)")

# Proxy rotator (shared across all downloads)
_proxy = ProxyRotator()


def _find_ytdlp() -> str:
    """Find yt-dlp binary — check venv bin first, then system PATH."""
    # Check same directory as the running Python interpreter (venv/bin/)
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    # Check if it's on system PATH
    found = shutil.which("yt-dlp")
    if found:
        return found
    raise FileNotFoundError(
        "yt-dlp not found. Install: pip install yt-dlp"
    )


# ─── Helpers ──────────────────────────────────────────────────


def _make_loader(proxy_str: str | None = None) -> instaloader.Instaloader:
    """Create an instaloader instance, optionally with a proxy."""
    loader = instaloader.Instaloader(
        download_comments=False,
        download_geotags=False,
        download_video_thumbnails=False,
        save_metadata=False,
        max_connection_attempts=3,
    )
    if proxy_str:
        loader.context._session.proxies = {
            "http": f"http://{proxy_str}",
            "https": f"http://{proxy_str}",
        }
    return loader


def _get_post(shortcode: str, max_proxy_attempts: int = 3) -> instaloader.Post:
    """Fetch post metadata via instaloader, rotating proxies on failure."""
    # First try direct (works on residential IPs)
    log.info("[instaloader] Direct fetch for %s...", shortcode)
    try:
        loader = _make_loader()
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        log.info("[instaloader] Direct fetch succeeded")
        return post
    except Exception as e:
        log.warning("[instaloader] Direct failed: %s — trying proxies", e)

    # Load proxy pool if needed
    if _proxy.count == 0:
        _proxy.refresh()

    for attempt in range(max_proxy_attempts):
        proxy_str = _proxy.get()
        if not proxy_str:
            break
        log.info("[instaloader] Proxy %d/%d: %s", attempt + 1, max_proxy_attempts, proxy_str)
        try:
            loader = _make_loader(proxy_str)
            post = instaloader.Post.from_shortcode(loader.context, shortcode)
            log.info("[instaloader] SUCCESS via %s", proxy_str)
            return post
        except Exception as e:
            log.warning("[instaloader] Proxy %s failed: %s", proxy_str, e)
            _proxy.remove(proxy_str)

    raise ConnectionError(
        f"instaloader failed for {shortcode} after {max_proxy_attempts} proxy attempts"
    )


def _extract_shortcode(url: str) -> str:
    """Extract shortcode from an Instagram URL."""
    m = _SHORTCODE_RE.search(url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract shortcode from URL: {url}")


# ─── Main Downloader ─────────────────────────────────────────


class ContentDownloader:
    """Downloads Instagram content with 2-level fallback.

    Fallback chain: instaloader → yt-dlp
    All methods use Proxifly free proxies for EC2 compatibility.
    """

    def __init__(self):
        Config.ensure_dirs()

    def download_url(self, url: str) -> tuple[MediaItem, DownloadedContent]:
        """Download content from an Instagram URL.

        Tries instaloader → yt-dlp.
        Returns (MediaItem, DownloadedContent) tuple.
        """
        shortcode = _extract_shortcode(url)
        post_dir = Config.MEDIA_DIR / shortcode
        post_dir.mkdir(parents=True, exist_ok=True)

        errors = []

        # === Method 1: instaloader + Proxifly (3 proxy attempts) ===
        try:
            log.info("[dl] === Method 1: instaloader for %s ===", shortcode)
            return self._via_instaloader(url, shortcode, post_dir)
        except Exception as e:
            errors.append(f"instaloader: {e}")
            log.warning("[dl] instaloader failed: %s", e)

        # === Method 2: yt-dlp + proxy ===
        try:
            log.info("[dl] === Method 2: yt-dlp for %s ===", shortcode)
            return self._via_ytdlp(url, shortcode, post_dir)
        except Exception as e:
            errors.append(f"yt-dlp: {e}")
            log.warning("[dl] yt-dlp failed: %s", e)

        log.error("[dl] ALL methods failed for %s: %s", shortcode, errors)

        # All failed
        item = MediaItem(
            media_id=shortcode, shortcode=shortcode,
            media_type=MediaType.IMAGE, url=url,
        )
        return item, DownloadedContent(media_item=item, success=False)

    def download(self, item: MediaItem, skip_keyframes: bool = False) -> DownloadedContent:
        """Download content for an existing MediaItem (used by pipeline).

        Uses same 3-level fallback chain.
        """
        post_dir = Config.MEDIA_DIR / item.shortcode
        post_dir.mkdir(parents=True, exist_ok=True)

        # === Method 1: instaloader ===
        try:
            log.info("[dl] Pipeline: instaloader for %s", item.shortcode)
            post = _get_post(item.shortcode)
            result = DownloadedContent(media_item=item, caption=item.caption)

            if item.media_type == MediaType.REEL:
                result = self._download_video(item.url, item.shortcode, post_dir, result)
                if not skip_keyframes and result.video_path and result.video_path.exists():
                    result.image_paths = self._extract_keyframes(result.video_path, post_dir)
                elif skip_keyframes:
                    log.info("Skipping keyframe extraction (Gemini will process video)")
            elif item.media_type == MediaType.CAROUSEL:
                result = self._download_carousel(post, post_dir, result)
            else:
                result = self._download_single_image(post, post_dir, result)

            result.success = len(result.image_paths) > 0 or result.video_path is not None
            if result.success:
                return result
            log.warning("[dl] instaloader got metadata but 0 files, trying fallback")
        except Exception as e:
            log.warning("[dl] instaloader pipeline failed: %s", e)

        # === Method 2: yt-dlp ===
        try:
            log.info("[dl] Pipeline fallback: yt-dlp for %s", item.shortcode)
            _, result = self._via_ytdlp(item.url, item.shortcode, post_dir)
            result.media_item = item
            if item.media_type == MediaType.REEL and not skip_keyframes and result.video_path:
                result.image_paths = self._extract_keyframes(result.video_path, post_dir)
            if result.success:
                return result
        except Exception as e:
            log.warning("[dl] yt-dlp pipeline failed: %s", e)

        log.error("[dl] ALL methods failed for pipeline download %s", item.shortcode)
        return DownloadedContent(media_item=item, success=False)

    # ─── Method 1: instaloader ────────────────────────────────

    def _via_instaloader(
        self, url: str, shortcode: str, post_dir: Path
    ) -> tuple[MediaItem, DownloadedContent]:
        """Download via instaloader with proxy rotation."""
        post = _get_post(shortcode)

        # Determine media type
        if post.is_video:
            media_type = MediaType.REEL
        elif post.typename == "GraphSidecar":
            media_type = MediaType.CAROUSEL
        else:
            media_type = MediaType.IMAGE

        item = MediaItem(
            media_id=shortcode, shortcode=shortcode,
            media_type=media_type, url=url,
            caption=post.caption or "",
            author_username=post.owner_username,
            timestamp=post.date_utc.isoformat() if post.date_utc else "",
        )
        result = DownloadedContent(media_item=item, caption=item.caption)

        if media_type == MediaType.REEL:
            result = self._download_video(url, shortcode, post_dir, result)
        elif media_type == MediaType.CAROUSEL:
            result = self._download_carousel(post, post_dir, result)
        else:
            result = self._download_single_image(post, post_dir, result)

        result.success = len(result.image_paths) > 0 or result.video_path is not None
        if not result.success:
            raise RuntimeError(f"instaloader got metadata but 0 files for {shortcode}")

        log.info(
            "[instaloader] Done: %d images, video=%s",
            len(result.image_paths), bool(result.video_path),
        )
        return item, result

    # ─── Method 2: yt-dlp ────────────────────────────────────

    def _via_ytdlp(
        self, url: str, shortcode: str, post_dir: Path
    ) -> tuple[MediaItem, DownloadedContent]:
        """Download via yt-dlp with proxy."""
        proxy_str = _proxy.get()
        proxy_args = ["--proxy", f"http://{proxy_str}"] if proxy_str else []

        # Step 1: Get metadata
        ytdlp = _find_ytdlp()
        meta_cmd = [ytdlp, "--dump-json"] + proxy_args + [url]
        proc = subprocess.run(meta_cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp metadata failed: {proc.stderr[:300]}")

        # Parse entries (carousel = multiple lines of JSON)
        lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        entries = [json.loads(l) for l in lines]
        first = entries[0]

        caption = first.get("description") or first.get("title") or ""
        author = first.get("uploader") or first.get("uploader_id") or ""
        timestamp = ""
        if first.get("timestamp"):
            timestamp = datetime.fromtimestamp(
                first["timestamp"], tz=timezone.utc
            ).isoformat()

        # Determine media type
        is_video = first.get("vcodec", "none") != "none"
        is_carousel = len(entries) > 1

        if is_video:
            media_type = MediaType.REEL
        elif is_carousel:
            media_type = MediaType.CAROUSEL
        else:
            media_type = MediaType.IMAGE

        item = MediaItem(
            media_id=shortcode, shortcode=shortcode,
            media_type=media_type, url=url,
            caption=caption, author_username=author,
            timestamp=timestamp,
        )
        result = DownloadedContent(media_item=item, caption=caption)

        # Step 2: Download content
        if media_type == MediaType.REEL:
            result = self._download_video(url, shortcode, post_dir, result)
        else:
            # Download image(s) — no --no-playlist so carousels get all slides
            output = str(post_dir / f"ytdlp_%(autonumber)s.%(ext)s")
            dl_cmd = [ytdlp, "-o", output] + proxy_args + [url]
            subprocess.run(dl_cmd, check=True, timeout=120, capture_output=True)

            # Collect downloaded files
            for f in sorted(post_dir.glob("ytdlp_*")):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    result.image_paths.append(f)
                elif f.suffix.lower() in (".mp4", ".webm"):
                    result.video_path = f
                    item.media_type = MediaType.REEL

        result.success = len(result.image_paths) > 0 or result.video_path is not None
        if not result.success:
            raise RuntimeError(f"yt-dlp downloaded 0 files for {shortcode}")

        log.info(
            "[yt-dlp] Done: %d images, video=%s",
            len(result.image_paths), bool(result.video_path),
        )
        return item, result

    # ─── Shared download helpers ─────────────────────────────

    def _download_single_image(
        self, post: instaloader.Post, post_dir: Path, result: DownloadedContent
    ) -> DownloadedContent:
        """Download a single image post via instaloader."""
        try:
            img_url = post.url
            r = requests.get(img_url, timeout=15)
            r.raise_for_status()
            path = post_dir / "image_0.jpg"
            path.write_bytes(r.content)
            result.image_paths.append(path)
            log.debug("Downloaded image: %s (%d bytes)", path, len(r.content))
        except Exception as e:
            log.error("Single image download failed: %s", e)
        return result

    def _download_carousel(
        self, post: instaloader.Post, post_dir: Path, result: DownloadedContent
    ) -> DownloadedContent:
        """Download all slides of a carousel post."""
        try:
            for i, node in enumerate(post.get_sidecar_nodes()):
                try:
                    img_url = node.display_url
                    r = requests.get(img_url, timeout=15)
                    r.raise_for_status()
                    path = post_dir / f"slide_{i}.jpg"
                    path.write_bytes(r.content)
                    result.image_paths.append(path)
                except Exception as e:
                    log.warning("Carousel slide %d download failed: %s", i, e)

            log.info("Downloaded %d carousel slides", len(result.image_paths))
        except Exception as e:
            log.error("Carousel download failed: %s", e)
        return result

    def _download_video(
        self, url: str, shortcode: str, post_dir: Path, result: DownloadedContent
    ) -> DownloadedContent:
        """Download a video (reel) via yt-dlp with retry + proxy rotation."""
        video_path = post_dir / f"{shortcode}.mp4"
        ytdlp = _find_ytdlp()

        for attempt in range(3):
            proxy_str = _proxy.get()
            cmd = [
                ytdlp,
                "-o", str(video_path),
                "--no-playlist",
                "--quiet",
                "--merge-output-format", "mp4",
            ]
            if proxy_str:
                cmd += ["--proxy", f"http://{proxy_str}"]
            cmd.append(url)

            try:
                subprocess.run(cmd, check=True, timeout=120)
                if video_path.exists():
                    result.video_path = video_path
                    log.info("Downloaded reel video: %s", video_path)
                    return result
            except Exception as e:
                if attempt < 2:
                    log.warning("Video download attempt %d/3 failed: %s", attempt + 1, e)
                    time.sleep(3)
                else:
                    log.error("Video download failed after 3 attempts: %s", e)

        return result

    def _extract_keyframes(self, video_path: Path, output_dir: Path) -> list[Path]:
        """Extract keyframes from video at configured interval using ffmpeg."""
        keyframes_dir = output_dir / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)

        interval = Config.REEL_KEYFRAME_INTERVAL
        output_pattern = str(keyframes_dir / "frame_%04d.jpg")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-vf", f"fps=1/{interval}",
                    "-q:v", "2",
                    "-y",
                    output_pattern,
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )

            frames = sorted(keyframes_dir.glob("frame_*.jpg"))
            log.info("Extracted %d keyframes (every %ds)", len(frames), interval)
            return frames

        except FileNotFoundError:
            log.error("ffmpeg not found! Install: brew install ffmpeg")
            return []
        except subprocess.TimeoutExpired:
            log.error("Keyframe extraction timed out")
            return []
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg failed: %s", e.stderr.decode() if e.stderr else str(e))
            return []
