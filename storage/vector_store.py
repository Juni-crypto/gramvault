"""Vector Store — ChromaDB for semantic search over content.

Indexes combined text (caption + OCR + description) as embeddings.
Supports semantic similarity search for the query interface.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from config import Config
from utils.logger import log


class VectorStore:
    """ChromaDB-backed vector store for semantic content search."""

    def __init__(self):
        Config.ensure_dirs()
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # Main collection: one document per post
        self.posts = self.client.get_or_create_collection(
            name="posts",
            metadata={"hnsw:space": "cosine"},
        )

        # Slides collection: individual carousel slides / keyframes
        self.slides = self.client.get_or_create_collection(
            name="slides",
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "Vector store initialized: %d posts, %d slides indexed",
            self.posts.count(),
            self.slides.count(),
        )

    def index_post(
        self,
        media_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """Index a post's combined text content."""
        if not text.strip():
            log.warning("Empty text for %s, skipping vector index", media_id)
            return

        meta = metadata or {}
        # ChromaDB metadata must be flat strings/ints/floats
        safe_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            elif isinstance(v, list):
                safe_meta[k] = ", ".join(str(x) for x in v)
            else:
                safe_meta[k] = str(v)

        self.posts.upsert(
            ids=[media_id],
            documents=[text[:10000]],  # ChromaDB document limit
            metadatas=[safe_meta],
        )
        log.debug("Indexed post %s (%d chars)", media_id, len(text))

    def index_slide(
        self,
        slide_id: str,
        media_id: str,
        text: str,
        slide_index: int = 0,
    ) -> None:
        """Index an individual slide/keyframe."""
        if not text.strip():
            return

        self.slides.upsert(
            ids=[slide_id],
            documents=[text[:10000]],
            metadatas={"media_id": media_id, "slide_index": slide_index},
        )

    def search(self, query: str, n_results: int = 10, collection: str = "posts") -> list[dict]:
        """Semantic search across indexed content.

        Returns list of dicts with: id, text, distance, metadata
        """
        col = self.posts if collection == "posts" else self.slides

        if col.count() == 0:
            return []

        results = col.query(
            query_texts=[query],
            n_results=min(n_results, col.count()),
        )

        items = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return items

    def reset(self):
        """Delete all collections and recreate them."""
        self.client.delete_collection("posts")
        self.client.delete_collection("slides")
        self.posts = self.client.get_or_create_collection(
            name="posts", metadata={"hnsw:space": "cosine"},
        )
        self.slides = self.client.get_or_create_collection(
            name="slides", metadata={"hnsw:space": "cosine"},
        )
        log.info("Vector store reset: all collections cleared")

    def get_stats(self) -> dict:
        return {
            "posts_indexed": self.posts.count(),
            "slides_indexed": self.slides.count(),
        }
