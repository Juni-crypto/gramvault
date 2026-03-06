"""Gemini Video Processor — Sends entire reels to Gemini for analysis.

Instead of extracting keyframes with ffmpeg and OCR-ing each frame,
we upload the whole video to Gemini's Files API and get:
  - Full text/OCR from video frames
  - Audio transcription (speech, music, narration)
  - Visual scene descriptions
  - Structured content extraction

~300 tokens/second of video at Gemini 2.0 Flash = ~$0.001 per 30s reel.
Compare to keyframe approach: ffmpeg + N separate Claude Vision calls.

Docs: https://ai.google.dev/gemini-api/docs/video-understanding
"""

import time
from pathlib import Path
from dataclasses import dataclass

from config import Config
from utils.logger import log


@dataclass
class VideoAnalysis:
    """Result of Gemini video analysis."""
    extracted_text: str = ""        # All text visible in the video
    audio_transcript: str = ""      # Speech/narration transcription
    description: str = ""           # What happens in the video
    key_topics: list[str] = None    # Topic tags
    tips: list[str] = None          # Actionable tips if any
    tokens_used: int = 0
    duration_seconds: float = 0
    success: bool = False

    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []
        if self.tips is None:
            self.tips = []

    @property
    def combined_text(self) -> str:
        """All extracted content combined for indexing."""
        parts = []
        if self.audio_transcript:
            parts.append(f"Audio/Speech: {self.audio_transcript}")
        if self.extracted_text:
            parts.append(f"On-screen Text: {self.extracted_text}")
        if self.description:
            parts.append(f"Visual Description: {self.description}")
        if self.tips:
            parts.append(f"Tips: {'; '.join(self.tips)}")
        return "\n\n".join(parts)


class GeminiVideoProcessor:
    """Processes videos using Gemini's native video understanding."""

    def __init__(self):
        self._client = None
        self._available = False
        self._total_tokens = 0
        self._total_videos = 0
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client."""
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            log.warning("GEMINI_API_KEY not set — Gemini video processing disabled")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._available = True
            log.info("Gemini video processor initialized (model: %s)", Config.GEMINI_VIDEO_MODEL)
        except ImportError:
            log.error("google-genai not installed. Run: pip install google-genai")
        except Exception as e:
            log.error("Gemini init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def process_video(self, video_path: Path) -> VideoAnalysis:
        """Upload video to Gemini and get full analysis.

        For reels under 20MB, uses inline data.
        For larger files, uses the Files API with upload + polling.
        """
        if not self._available:
            log.warning("Gemini not available, returning empty analysis")
            return VideoAnalysis()

        if not video_path.exists():
            log.error("Video not found: %s", video_path)
            return VideoAnalysis()

        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        log.info("Processing video: %s (%.1f MB)", video_path.name, file_size_mb)

        try:
            if file_size_mb < Config.GEMINI_INLINE_MAX_MB:
                return self._process_inline(video_path)
            else:
                return self._process_with_upload(video_path)
        except Exception as e:
            log.error("Gemini video processing failed: %s", e)
            return VideoAnalysis()

    def _process_inline(self, video_path: Path) -> VideoAnalysis:
        """Process small videos (<20MB) with inline data."""
        from google.genai import types

        video_bytes = video_path.read_bytes()

        # Determine MIME type
        suffix = video_path.suffix.lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
        }
        mime_type = mime_map.get(suffix, "video/mp4")

        response = self._client.models.generate_content(
            model=Config.GEMINI_VIDEO_MODEL,
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
                self._build_prompt(),
            ],
        )

        return self._parse_response(response)

    def _process_with_upload(self, video_path: Path) -> VideoAnalysis:
        """Process larger videos using Gemini Files API (upload → poll → analyze)."""
        log.info("Uploading video to Gemini Files API...")

        # Upload
        uploaded_file = self._client.files.upload(file=str(video_path))
        log.info("Uploaded: %s (state: %s)", uploaded_file.name, uploaded_file.state)

        # Poll until processing is complete
        max_wait = Config.GEMINI_UPLOAD_TIMEOUT
        waited = 0
        poll_interval = 5
        while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval
            uploaded_file = self._client.files.get(name=uploaded_file.name)
            log.debug("File state: %s (waited %ds)", uploaded_file.state.name, waited)

        if uploaded_file.state.name == "FAILED":
            log.error("Gemini file processing failed")
            return VideoAnalysis()

        if uploaded_file.state.name != "ACTIVE":
            log.error("File not ready after %ds (state: %s)", max_wait, uploaded_file.state.name)
            return VideoAnalysis()

        log.info("File ready, generating analysis...")

        response = self._client.models.generate_content(
            model=Config.GEMINI_VIDEO_MODEL,
            contents=[uploaded_file, self._build_prompt()],
        )

        # Clean up uploaded file
        try:
            self._client.files.delete(name=uploaded_file.name)
        except Exception:
            pass  # Non-critical

        return self._parse_response(response)

    def _build_prompt(self) -> str:
        """Build the analysis prompt for Gemini."""
        return (
            "Analyze this Instagram reel video thoroughly. Extract ALL information:\n\n"
            "1. AUDIO_TRANSCRIPT: Transcribe ALL spoken words, narration, or voiceover. "
            "Include song lyrics if relevant to the content.\n\n"
            "2. ON_SCREEN_TEXT: Extract ALL text that appears on screen — "
            "titles, subtitles, captions, bullet points, labels, watermarks, usernames.\n\n"
            "3. VISUAL_DESCRIPTION: Describe what happens visually — "
            "people, actions, products shown, locations, demonstrations, "
            "before/after comparisons, tutorials steps.\n\n"
            "4. KEY_TOPICS: List 3-7 topic tags for this content.\n\n"
            "5. TIPS: If the video contains advice, tips, tutorials, or "
            "how-to information, list each actionable tip separately.\n\n"
            "Format your response EXACTLY as:\n"
            "AUDIO_TRANSCRIPT:\n<transcription>\n\n"
            "ON_SCREEN_TEXT:\n<text>\n\n"
            "VISUAL_DESCRIPTION:\n<description>\n\n"
            "KEY_TOPICS:\n<comma-separated topics>\n\n"
            "TIPS:\n- <tip 1>\n- <tip 2>\n..."
        )

    def _parse_response(self, response) -> VideoAnalysis:
        """Parse Gemini's structured response into VideoAnalysis."""
        try:
            text = response.text
            self._total_videos += 1

            # Track tokens
            usage = getattr(response, "usage_metadata", None)
            tokens = 0
            if usage:
                tokens = getattr(usage, "total_token_count", 0)
                self._total_tokens += tokens

            analysis = VideoAnalysis(
                audio_transcript=self._extract_section(text, "AUDIO_TRANSCRIPT"),
                extracted_text=self._extract_section(text, "ON_SCREEN_TEXT"),
                description=self._extract_section(text, "VISUAL_DESCRIPTION"),
                key_topics=self._extract_list(text, "KEY_TOPICS"),
                tips=self._extract_tips(text, "TIPS"),
                tokens_used=tokens,
                success=True,
            )

            log.info(
                "Video analysis complete: %d chars transcript, %d chars text, %d tips, %d tokens",
                len(analysis.audio_transcript),
                len(analysis.extracted_text),
                len(analysis.tips),
                tokens,
            )
            return analysis

        except Exception as e:
            log.error("Failed to parse Gemini response: %s", e)
            # Return raw text as fallback
            try:
                return VideoAnalysis(
                    description=response.text[:5000],
                    success=True,
                )
            except Exception:
                return VideoAnalysis()

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """Extract a named section from structured response."""
        try:
            sections = [
                "AUDIO_TRANSCRIPT", "ON_SCREEN_TEXT",
                "VISUAL_DESCRIPTION", "KEY_TOPICS", "TIPS",
            ]
            start_marker = f"{section_name}:"
            if start_marker not in text:
                return ""

            content = text.split(start_marker, 1)[1]

            # Find next section
            for next_sec in sections:
                if next_sec != section_name:
                    marker = f"{next_sec}:"
                    if marker in content:
                        content = content.split(marker)[0]

            return content.strip()
        except Exception:
            return ""

    @staticmethod
    def _extract_list(text: str, section_name: str) -> list[str]:
        """Extract comma-separated list from a section."""
        section = GeminiVideoProcessor._extract_section(text, section_name)
        if not section:
            return []
        items = [item.strip().strip("#").strip() for item in section.split(",")]
        return [i for i in items if i and len(i) > 1]

    @staticmethod
    def _extract_tips(text: str, section_name: str) -> list[str]:
        """Extract bullet-pointed tips from a section."""
        section = GeminiVideoProcessor._extract_section(text, section_name)
        if not section:
            return []
        tips = []
        for line in section.split("\n"):
            line = line.strip().lstrip("-•*").strip()
            if line and len(line) > 5:
                tips.append(line)
        return tips

    def get_cost_summary(self) -> dict:
        """Cost tracking for Gemini video processing."""
        # Gemini 2.0 Flash pricing
        cost_per_million_input = 0.10  # $0.10/1M input tokens
        return {
            "total_videos": self._total_videos,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": (self._total_tokens / 1_000_000) * cost_per_million_input,
            "provider": f"gemini ({Config.GEMINI_VIDEO_MODEL})",
        }
