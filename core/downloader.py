"""Content Downloader — Downloads Instagram content.

Uses instaloader for metadata + images, yt-dlp for videos (reels).
Auto-rotates free proxies on datacenter IPs (EC2).

Handles:
- Single image posts → downloads image
- Carousels → downloads all slides
- Reels → downloads video + extracts keyframes via ffmpeg
"""

import re
import subprocess
import time
from dataclasses import dataclass, field
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


def _retry(func, max_retries=2, delay=3):
    """Simple retry wrapper with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise
            log.warning("Retry %d/%d after error: %s", attempt + 1, max_retries, e)
            time.sleep(delay * (attempt + 1))


_SHORTCODE_RE = re.compile(r"instagram\.com/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)")

# Proxy rotator (shared across all downloads)
_proxy = ProxyRotator()


def _make_loader(proxy_str: str | None = None) -> instaloader.Instaloader:
    """Create an instaloader instance, optionally with a proxy."""
    loader = instaloader.Instaloader(
        download_comments=False,
        download_geotags=False,
        download_video_thumbnails=False,
        save_metadata=False,
        max_connection_attempts=1,  # Don't retry internally — we handle rotation
    )
    if proxy_str:
        loader.context._session.proxies = {
            "http": f"http://{proxy_str}",
            "https": f"http://{proxy_str}",
        }
    return loader


def _get_post(shortcode: str, max_proxy_attempts: int = 5) -> instaloader.Post:
    """Fetch post metadata, rotating proxies on failure."""
    # First try without proxy (works on residential IPs)
    log.info("[downloader] Attempting direct fetch for %s...", shortcode)
    try:
        loader = _make_loader()
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        log.info("[downloader] Direct fetch succeeded for %s", shortcode)
        return post
    except Exception as first_err:
        log.warning("[downloader] Direct fetch failed: %s — switching to proxies", first_err)

    # Load proxy pool if empty
    if _proxy.count == 0:
        log.info("[downloader] Proxy pool empty, fetching proxies...")
        _proxy.refresh()
        log.info("[downloader] Proxy pool loaded: %d proxies", _proxy.count)

    # Rotate through proxies
    for attempt in range(max_proxy_attempts):
        proxy_str = _proxy.get()
        if not proxy_str:
            log.warning("[downloader] No proxies available (attempt %d/%d)", attempt + 1, max_proxy_attempts)
            break

        log.info("[downloader] Trying proxy %d/%d: %s", attempt + 1, max_proxy_attempts, proxy_str)
        try:
            loader = _make_loader(proxy_str)
            post = instaloader.Post.from_shortcode(loader.context, shortcode)
            log.info("[downloader] SUCCESS via proxy %s", proxy_str)
            return post
        except Exception as e:
            log.warning("[downloader] Proxy %s failed: %s", proxy_str, e)
            _proxy.remove(proxy_str)

    raise ConnectionError(
        f"Could not fetch {shortcode} — direct and {max_proxy_attempts} proxies all failed. "
        f"Proxy pool: {_proxy.count} remaining"
    )


def _extract_shortcode(url: str) -> str:
    """Extract shortcode from an Instagram URL."""
    m = _SHORTCODE_RE.search(url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract shortcode from URL: {url}")


class ContentDownloader:
    """Downloads Instagram content using instaloader + yt-dlp."""

    def __init__(self):
        Config.ensure_dirs()

    def download_url(self, url: str) -> tuple[MediaItem, DownloadedContent]:
        """Download content from an Instagram URL.

        Returns (MediaItem, DownloadedContent) tuple.
        """
        shortcode = _extract_shortcode(url)
        post_dir = Config.MEDIA_DIR / shortcode
        post_dir.mkdir(parents=True, exist_ok=True)

        # Get metadata via instaloader (with proxy rotation)
        post = _get_post(shortcode)

        # Determine media type
        if post.is_video:
            media_type = MediaType.REEL
        elif post.typename == "GraphSidecar":
            media_type = MediaType.CAROUSEL
        else:
            media_type = MediaType.IMAGE

        item = MediaItem(
            media_id=shortcode,
            shortcode=shortcode,
            media_type=media_type,
            url=url,
            caption=post.caption or "",
            author_username=post.owner_username,
            timestamp=post.date_utc.isoformat() if post.date_utc else "",
        )

        result = DownloadedContent(media_item=item, caption=item.caption)

        try:
            if media_type == MediaType.REEL:
                result = self._download_video(url, shortcode, post_dir, result)
            elif media_type == MediaType.CAROUSEL:
                result = self._download_carousel(post, post_dir, result)
            else:
                result = self._download_single_image(post, post_dir, result)

            result.success = len(result.image_paths) > 0 or result.video_path is not None
            log.info(
                "Downloaded %s: %d images, video=%s",
                shortcode,
                len(result.image_paths),
                "yes" if result.video_path else "no",
            )
        except Exception as e:
            log.error("Download failed for %s: %s", shortcode, e)
            result.success = False

        return item, result

    def download(self, item: MediaItem, skip_keyframes: bool = False) -> DownloadedContent:
        """Download content for an existing MediaItem (used by pipeline)."""
        result = DownloadedContent(media_item=item, caption=item.caption)
        post_dir = Config.MEDIA_DIR / item.shortcode
        post_dir.mkdir(parents=True, exist_ok=True)

        try:
            post = _get_post(item.shortcode)

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
        except Exception as e:
            log.error("Download failed for %s: %s", item.shortcode, e)
            result.success = False

        return result

    def _download_single_image(
        self, post: instaloader.Post, post_dir: Path, result: DownloadedContent
    ) -> DownloadedContent:
        """Download a single image post."""
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
        """Download a video (reel) via yt-dlp."""
        video_path = post_dir / f"{shortcode}.mp4"

        def _do_download():
            cmd = [
                "yt-dlp",
                "-o", str(video_path),
                "--no-playlist",
                "--quiet",
                "--merge-output-format", "mp4",
            ]
            # Use proxy if available
            proxy_str = _proxy.get()
            if proxy_str:
                cmd += ["--proxy", f"http://{proxy_str}"]
            cmd.append(url)
            subprocess.run(cmd, check=True, timeout=120)

        try:
            _retry(_do_download)
            if video_path.exists():
                result.video_path = video_path
                log.info("Downloaded reel video: %s", video_path)
        except Exception as e:
            log.error("Video download failed: %s", e)

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
