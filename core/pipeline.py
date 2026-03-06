"""Processing Pipeline — Orchestrates the full content intelligence flow.

Instagram URL → Download Media → OCR/Vision → Entity Extraction → Index → Graph
"""

from datetime import datetime, timezone

from core.models import MediaItem, MediaType
from core.downloader import ContentDownloader, DownloadedContent
from core.vision import VisionProcessor
from core.gemini_video import GeminiVideoProcessor, VideoAnalysis
from core.entity_extractor import EntityExtractor
from storage.database import Database
from storage.vector_store import VectorStore
from storage.knowledge_graph import KnowledgeGraph
from utils.logger import log


class Pipeline:
    """Orchestrates the complete processing pipeline."""

    def __init__(
        self,
        downloader: ContentDownloader,
        vision: VisionProcessor,
        gemini_video: GeminiVideoProcessor,
        extractor: EntityExtractor,
        db: Database,
        vectors: VectorStore,
        graph: KnowledgeGraph,
    ):
        self.downloader = downloader
        self.vision = vision
        self.gemini_video = gemini_video
        self.extractor = extractor
        self.db = db
        self.vectors = vectors
        self.graph = graph

        self._processed_count = 0
        self._error_count = 0

    def process_url(self, url: str) -> dict:
        """Process an Instagram URL through the full pipeline.

        Returns a summary dict with category, author, topics, tips, summary.
        """
        # Step 1: Download and extract metadata
        log.info("Processing URL: %s", url)
        item, downloaded = self.downloader.download_url(url)

        # Dedup check
        existing = self.db.get_post(item.shortcode)
        if existing and existing.get("status") == "done":
            log.info("Already processed: %s", item.shortcode)
            return {
                "status": "duplicate",
                "shortcode": item.shortcode,
                "summary": existing.get("summary", ""),
                "category": existing.get("category", ""),
                "author": existing.get("author_username", ""),
            }

        if not downloaded.success:
            log.error("Download failed for %s", item.shortcode)
            return {"status": "error", "error": "Download failed"}

        # Process through pipeline
        self.process_item(item, downloaded)

        # Return summary
        post = self.db.get_post(item.shortcode) or {}
        return {
            "status": "processed",
            "shortcode": item.shortcode,
            "summary": post.get("summary", ""),
            "category": post.get("category", "general"),
            "author": item.author_username,
            "url": item.url,
        }

    def process_item(self, item: MediaItem, downloaded: DownloadedContent | None = None) -> None:
        """Process a single media item through the full pipeline."""
        log.info("━" * 60)
        log.info("Processing: %s (%s) by @%s", item.shortcode, item.media_type.value, item.author_username)

        # Step 1: Create initial DB record
        self.db.upsert_post(
            media_id=item.media_id,
            shortcode=item.shortcode,
            media_type=item.media_type.value,
            url=item.url,
            caption=item.caption,
            author_username=item.author_username,
            author_id=item.author_id,
            post_timestamp=item.timestamp,
            hashtags=item.hashtags,
            status="downloading",
        )

        # Step 2: Download content (if not already downloaded)
        if downloaded is None:
            log.info("Step 2/5: Downloading...")
            skip_kf = (item.media_type == MediaType.REEL and self.gemini_video.available)
            downloaded = self.downloader.download(item, skip_keyframes=skip_kf)

        if not downloaded.success:
            log.error("Download failed for %s, marking as error", item.shortcode)
            self.db.upsert_post(media_id=item.media_id, status="download_error")
            return

        self.db.upsert_post(media_id=item.media_id, status="processing_vision")

        # Step 3: Extract content — different paths for reels vs images
        combined_ocr = ""
        combined_desc = ""
        extra_topics = []
        extra_tips = []

        if item.media_type == MediaType.REEL and self.gemini_video.available and downloaded.video_path:
            log.info("Step 3/5: Gemini video analysis (whole reel)...")
            analysis = self.gemini_video.process_video(downloaded.video_path)

            if analysis.success:
                combined_ocr = f"{analysis.audio_transcript}\n\n{analysis.extracted_text}"
                combined_desc = analysis.description
                extra_topics = analysis.key_topics
                extra_tips = analysis.tips

                log.info(
                    "Gemini reel analysis: %d chars transcript, %d chars text, %d tips",
                    len(analysis.audio_transcript), len(analysis.extracted_text), len(analysis.tips),
                )
            else:
                log.warning("Gemini failed, falling back to keyframe OCR...")
                combined_ocr, combined_desc = self._process_images(
                    item.media_id, downloaded.image_paths
                )
        else:
            log.info(
                "Step 3/5: Vision/OCR processing (%d images)...",
                len(downloaded.image_paths),
            )
            combined_ocr, combined_desc = self._process_images(
                item.media_id, downloaded.image_paths
            )

        self.db.upsert_post(
            media_id=item.media_id,
            ocr_text=combined_ocr,
            description=combined_desc,
            status="extracting_entities",
        )

        # Step 4: Entity extraction
        log.info("Step 4/5: Extracting entities...")
        entities = self.extractor.extract(
            caption=item.caption,
            ocr_text=combined_ocr,
            description=combined_desc,
        )

        if extra_topics:
            entities.topics = list(set(entities.topics + extra_topics))
        if extra_tips:
            entities.tips = list(set(entities.tips + extra_tips))

        self.db.upsert_post(
            media_id=item.media_id,
            category=entities.category,
            summary=entities.summary,
            entities_json={
                "topics": entities.topics,
                "people": entities.people,
                "brands": entities.brands,
                "products": entities.products,
                "tips": entities.tips,
                "locations": entities.locations,
                "key_facts": entities.key_facts,
            },
            status="indexing",
        )

        # Step 5: Index in vector store + knowledge graph
        log.info("Step 5/5: Indexing...")

        full_text = (
            f"Caption: {item.caption}\n\n"
            f"Content Text: {combined_ocr}\n\n"
            f"Description: {combined_desc}\n\n"
            f"Summary: {entities.summary}\n\n"
            f"Tips: {'; '.join(entities.tips)}\n\n"
            f"Key Facts: {'; '.join(entities.key_facts)}"
        )

        self.vectors.index_post(
            media_id=item.media_id,
            text=full_text,
            metadata={
                "shortcode": item.shortcode,
                "author": item.author_username,
                "category": entities.category,
                "media_type": item.media_type.value,
                "url": item.url,
            },
        )

        self.graph.add_post(
            media_id=item.media_id,
            shortcode=item.shortcode,
            author=item.author_username,
            entities=entities,
            url=item.url,
        )

        # Mark complete
        self.db.upsert_post(
            media_id=item.media_id,
            status="done",
            processed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.db.mark_url_processed(item.shortcode)

        self._processed_count += 1
        log.info(
            "Done: %s | category=%s | %d topics, %d tips | %s",
            item.shortcode,
            entities.category,
            len(entities.topics),
            len(entities.tips),
            entities.summary[:80],
        )

    def _process_images(self, media_id: str, image_paths: list) -> tuple[str, str]:
        """Process images through Vision/OCR pipeline. Returns (ocr_text, description)."""
        all_ocr = []
        all_desc = []

        vision_results = self.vision.process_images(image_paths)
        for i, vr in enumerate(vision_results):
            if vr.success:
                all_ocr.append(vr.extracted_text)
                all_desc.append(vr.description)

                self.db.add_slide(
                    media_id=media_id,
                    slide_index=i,
                    image_path=str(image_paths[i]),
                    ocr_text=vr.extracted_text,
                    description=vr.description,
                    provider=vr.provider,
                )

                slide_id = f"{media_id}_slide_{i}"
                self.vectors.index_slide(
                    slide_id=slide_id,
                    media_id=media_id,
                    text=f"{vr.extracted_text}\n{vr.description}",
                    slide_index=i,
                )

        return "\n---\n".join(all_ocr), "\n".join(all_desc)

    def get_stats(self) -> dict:
        return {
            "pipeline": {
                "processed": self._processed_count,
                "errors": self._error_count,
            },
            "database": self.db.get_stats(),
            "vectors": self.vectors.get_stats(),
            "graph": self.graph.get_stats(),
            "vision": self.vision.get_cost_summary(),
        }
