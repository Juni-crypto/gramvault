"""Vision / OCR Processor — Extracts text and descriptions from images.

Provider priority:
  1. Gemini (fast, supports batching all images in one call)
  2. Claude Vision API
  3. Tesseract OCR (fallback)
"""

import base64
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import Config
from utils.logger import log


@dataclass
class VisionResult:
    """Result from processing a single image."""
    extracted_text: str = ""
    description: str = ""
    provider: str = "none"
    tokens_used: int = 0
    success: bool = False


def auto_detect_provider() -> str:
    """Determine the best available vision provider."""
    override = Config.VISION_PROVIDER.lower()
    if override != "auto":
        log.info("Vision provider override: %s", override)
        return override

    if Config.GEMINI_API_KEY:
        log.info("Gemini API key found. Using Gemini Vision.")
        return "gemini"

    if Config.ANTHROPIC_API_KEY:
        log.info("Anthropic API key found. Using Claude Vision.")
        return "claude"

    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            log.info("Falling back to Tesseract OCR.")
            return "tesseract"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    log.warning("No vision provider available!")
    return "none"


class VisionProcessor:
    """Processes images through the detected vision provider."""

    def __init__(self):
        self.provider = auto_detect_provider()
        self._total_tokens = 0
        self._total_images = 0
        self._gemini_client = None

        if self.provider == "gemini":
            self._init_gemini()

        log.info("Vision processor initialized: provider=%s", self.provider)

    def _init_gemini(self):
        """Initialize Gemini client for image processing."""
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)
        except Exception as e:
            log.error("Gemini vision init failed: %s", e)
            self.provider = "tesseract"

    def process_images(self, image_paths: list[Path]) -> list[VisionResult]:
        """Process multiple images — batched for Gemini, sequential for others."""
        if not image_paths:
            return []

        if self.provider == "gemini" and self._gemini_client:
            return self._process_gemini_batch(image_paths)

        # Sequential fallback for claude/tesseract
        results = []
        for i, path in enumerate(image_paths):
            log.info("Processing image %d/%d: %s", i + 1, len(image_paths), path.name)
            result = self.process_image(path)
            results.append(result)
        return results

    def process_image(self, image_path: Path) -> VisionResult:
        """Process a single image."""
        if not image_path.exists():
            log.warning("Image not found: %s", image_path)
            return VisionResult()

        if self.provider == "gemini":
            results = self._process_gemini_batch([image_path])
            return results[0] if results else VisionResult()
        elif self.provider == "claude":
            return self._process_claude(image_path)
        elif self.provider == "tesseract":
            return self._process_tesseract(image_path)
        else:
            return VisionResult()

    def get_cost_summary(self) -> dict:
        return {
            "total_images": self._total_images,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": (self._total_tokens / 1_000_000) * 0.10,
            "provider": self.provider,
        }

    # ─── Gemini Vision (batched) ───────────────────────────────────

    def _process_gemini_batch(self, image_paths: list[Path]) -> list[VisionResult]:
        """Send all images to Gemini in a single API call."""
        from google.genai import types

        log.info("Gemini batch processing %d images...", len(image_paths))

        parts = []
        valid_paths = []
        for path in image_paths:
            if not path.exists():
                continue
            try:
                img_bytes = path.read_bytes()
                suffix = path.suffix.lower()
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                        ".webp": "image/webp", ".gif": "image/gif"}.get(suffix, "image/jpeg")
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                valid_paths.append(path)
            except Exception as e:
                log.warning("Skipping %s: %s", path.name, e)

        if not parts:
            return [VisionResult() for _ in image_paths]

        # Build prompt
        if len(parts) == 1:
            prompt_text = (
                "Analyze this Instagram post image. Provide:\n"
                "EXTRACTED_TEXT: All text visible in the image\n"
                "DESCRIPTION: What the image shows\n\n"
                "Format as:\nEXTRACTED_TEXT:\n<text>\n\nDESCRIPTION:\n<description>"
            )
        else:
            prompt_text = (
                f"Analyze these {len(parts)} Instagram carousel images. "
                f"For EACH image (numbered 1 to {len(parts)}), provide:\n"
                "- EXTRACTED_TEXT: All text visible\n"
                "- DESCRIPTION: What the image shows\n\n"
                "Format as:\n"
                "IMAGE 1:\nEXTRACTED_TEXT:\n<text>\nDESCRIPTION:\n<description>\n\n"
                "IMAGE 2:\nEXTRACTED_TEXT:\n<text>\nDESCRIPTION:\n<description>\n..."
            )

        parts.append(prompt_text)

        try:
            response = self._gemini_client.models.generate_content(
                model=Config.GEMINI_VIDEO_MODEL,
                contents=parts,
            )

            text = response.text
            usage = getattr(response, "usage_metadata", None)
            tokens = getattr(usage, "total_token_count", 0) if usage else 0
            self._total_tokens += tokens
            self._total_images += len(valid_paths)

            log.info("Gemini batch done: %d images, %d tokens", len(valid_paths), tokens)

            # Parse per-image results
            if len(valid_paths) == 1:
                extracted = self._parse_section(text, "EXTRACTED_TEXT")
                description = self._parse_section(text, "DESCRIPTION")
                return [VisionResult(
                    extracted_text=extracted, description=description,
                    provider="gemini", tokens_used=tokens, success=True,
                )]

            return self._parse_batch_response(text, len(valid_paths), tokens)

        except Exception as e:
            log.error("Gemini batch failed: %s", e)
            results = []
            for path in image_paths:
                results.append(self._process_tesseract(path))
            return results

    def _parse_batch_response(self, text: str, count: int, total_tokens: int) -> list[VisionResult]:
        """Parse Gemini's multi-image response into per-image results."""
        results = []
        tokens_per = total_tokens // max(count, 1)

        for i in range(1, count + 1):
            marker = f"IMAGE {i}:"
            next_marker = f"IMAGE {i + 1}:"

            if marker in text:
                section = text.split(marker, 1)[1]
                if next_marker in section:
                    section = section.split(next_marker)[0]

                extracted = self._parse_section(section, "EXTRACTED_TEXT")
                description = self._parse_section(section, "DESCRIPTION")
            else:
                extracted = ""
                description = ""

            results.append(VisionResult(
                extracted_text=extracted,
                description=description,
                provider="gemini",
                tokens_used=tokens_per,
                success=bool(extracted or description),
            ))

        while len(results) < count:
            results.append(VisionResult(provider="gemini"))

        return results

    # ─── Claude Vision API ─────────────────────────────────────────

    def _process_claude(self, image_path: Path) -> VisionResult:
        """Use Claude Vision API to extract text and describe image."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            img_bytes = image_path.read_bytes()
            b64_data = base64.b64encode(img_bytes).decode("utf-8")

            suffix = image_path.suffix.lower()
            media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                          ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

            response = client.messages.create(
                model=Config.ANTHROPIC_CHAT_MODEL,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                        {"type": "text", "text": (
                            "Analyze this Instagram post image. Provide:\n"
                            "EXTRACTED_TEXT: All text visible in the image\n"
                            "DESCRIPTION: What the image shows\n\n"
                            "Format as:\nEXTRACTED_TEXT:\n<text>\n\nDESCRIPTION:\n<description>"
                        )},
                    ],
                }],
            )

            text = response.content[0].text
            self._total_tokens += response.usage.input_tokens
            self._total_images += 1

            return VisionResult(
                extracted_text=self._parse_section(text, "EXTRACTED_TEXT"),
                description=self._parse_section(text, "DESCRIPTION"),
                provider="claude",
                tokens_used=response.usage.input_tokens,
                success=True,
            )
        except Exception as e:
            log.error("Claude Vision failed for %s: %s", image_path.name, e)
            return self._process_tesseract(image_path)

    # ─── Tesseract OCR ─────────────────────────────────────────────

    def _process_tesseract(self, image_path: Path) -> VisionResult:
        """Use Tesseract for basic OCR text extraction."""
        try:
            import pytesseract
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang="eng").strip()
            self._total_images += 1
            return VisionResult(
                extracted_text=text,
                description="[Tesseract OCR — no image description available]",
                provider="tesseract", tokens_used=0, success=bool(text),
            )
        except Exception as e:
            log.error("Tesseract OCR failed for %s: %s", image_path.name, e)
            return VisionResult(provider="tesseract")

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_section(text: str, section_name: str) -> str:
        try:
            parts = text.split(f"{section_name}:")
            if len(parts) < 2:
                return ""
            content = parts[1]
            for header in ["EXTRACTED_TEXT:", "DESCRIPTION:", "KEY_TOPICS:", "IMAGE "]:
                if header != f"{section_name}:" and header in content:
                    content = content.split(header)[0]
            return content.strip()
        except Exception:
            return ""
