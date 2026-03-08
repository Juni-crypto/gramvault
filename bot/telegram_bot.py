"""Telegram Bot — Claude-powered RAG chat + Instagram URL processing.

Send an Instagram URL -> queued, processed one per minute, indexed.
Ask any question -> Claude searches your saved content and answers.

Commands:
    /start      — Welcome
    /stats      — Pipeline statistics
    /topics     — Top topics in your graph
    /recent     — Latest saved posts
    /category   — Browse categories
    /graph      — Knowledge graph visualization
    /cost       — AI API usage & costs
    /queue      — Check URL processing queue
    /flush      — Delete all saved data
    <URL>       — Queue an Instagram post/reel for processing
    <any text>  — Claude-powered search & answer
"""

import asyncio
import collections
import json
import re
import shutil
from datetime import time as dt_time, timezone, timedelta

import anthropic
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import Config
from core.downloader import ContentDownloader
from core.vision import VisionProcessor
from core.gemini_video import GeminiVideoProcessor
from core.entity_extractor import EntityExtractor
from core.pipeline import Pipeline
from storage.database import Database
from storage.vector_store import VectorStore
from storage.knowledge_graph import KnowledgeGraph
from utils.logger import log


# ─── Shared state ─────────────────────────────────────────────
_db: Database | None = None
_vectors: VectorStore | None = None
_graph: KnowledgeGraph | None = None
_claude: anthropic.Anthropic | None = None
_pipeline: Pipeline | None = None

# RAG chat cost tracking
_rag_input_tokens = 0
_rag_output_tokens = 0
_rag_calls = 0

_INSTAGRAM_URL_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:p|reel|reels|tv)/[A-Za-z0-9_-]+"
)

_SHORTCODE_RE = re.compile(
    r"instagram\.com/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)"
)

# IST = UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))

# ─── URL Processing Queue ─────────────────────────────────
# Each item: (url, chat_id, user_id)
_url_queue: collections.deque = collections.deque()

# Processing stats for daily digest (reset nightly)
_processing_stats = {
    "processed": 0,
    "failed": 0,
    "errors": [],   # list of {"url": str, "reason": str}
}


def _check_authorized(user_id: int) -> bool:
    if not Config.TELEGRAM_ALLOWED_USERS:
        log.warning("TELEGRAM_ALLOWED_USERS is empty -- denying access to user %s", user_id)
        return False
    return user_id in Config.TELEGRAM_ALLOWED_USERS


# ─── Claude RAG Engine ────────────────────────────────────────

def _build_context_from_results(results: list[dict]) -> str:
    if not results:
        return "No saved content found matching this query."

    context_parts = []
    for i, r in enumerate(results):
        post = r.get("post", {})
        author = post.get("author_username", "unknown")
        category = post.get("category", "general")
        media_type = post.get("media_type", "")
        url = post.get("url", "")
        caption = post.get("caption", "")[:500]
        ocr_text = post.get("ocr_text", "")[:800]
        description = post.get("description", "")[:400]
        summary = post.get("summary", "")

        tips, key_facts = [], []
        try:
            entities = json.loads(post.get("entities_json", "{}"))
            tips = entities.get("tips", [])
            key_facts = entities.get("key_facts", [])
        except (json.JSONDecodeError, TypeError):
            pass

        part = f"--- SAVED POST #{i+1} ---\n"
        part += f"Author: @{author} | Type: {media_type} | Category: {category}\n"
        part += f"URL: {url}\n"
        if summary:
            part += f"Summary: {summary}\n"
        if caption:
            part += f"Caption: {caption}\n"
        if ocr_text:
            part += f"Content Text: {ocr_text}\n"
        if description:
            part += f"Visual Description: {description}\n"
        if tips:
            part += f"Tips: {'; '.join(tips)}\n"
        if key_facts:
            part += f"Key Facts: {'; '.join(key_facts)}\n"
        context_parts.append(part)

    return "\n".join(context_parts)


def _ask_claude(query: str, context: str) -> str:
    global _rag_input_tokens, _rag_output_tokens, _rag_calls
    if not _claude:
        return "Claude API not configured. Set ANTHROPIC_API_KEY in .env"

    try:
        response = _claude.messages.create(
            model=Config.ANTHROPIC_CHAT_MODEL,
            max_tokens=1500,
            system=(
                "You are InstaIntel, an AI assistant that helps users recall and "
                "find information from Instagram posts they've saved. You have access "
                "to the user's saved content provided below as context.\n\n"
                "RULES:\n"
                "- Answer based ONLY on the provided saved content. Don't make up info.\n"
                "- If the content has relevant tips, share them clearly.\n"
                "- Always mention which post/author the info came from.\n"
                "- Include the Instagram URL so the user can revisit the original.\n"
                "- Be concise and conversational — this is a Telegram chat.\n"
                "- If nothing matches, say so honestly and suggest the user try "
                "different keywords or check /topics for what's available.\n"
                "- Use emoji sparingly for readability.\n"
                "- Format tips as a clean numbered list when applicable."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Here are my saved Instagram posts relevant to my question:\n\n"
                    f"{context}\n\n---\n\nMy question: {query}"
                ),
            }],
        )
        _rag_input_tokens += response.usage.input_tokens
        _rag_output_tokens += response.usage.output_tokens
        _rag_calls += 1
        return response.content[0].text
    except anthropic.RateLimitError:
        return "Rate limited. Try again in a moment."
    except Exception as e:
        log.error("Claude RAG failed: %s", e)
        return "Error generating response. Try /recent to browse."


def _fallback_response(results: list[dict]) -> str:
    if not results:
        return "No results found. Try different keywords or /topics."
    lines = []
    for i, r in enumerate(results[:5]):
        post = r.get("post", {})
        author = post.get("author_username", "?")
        summary = post.get("summary", "")[:120]
        url = post.get("url", "")
        lines.append(f"#{i+1} @{author}: {summary}\n{url}")
    return "Results:\n\n" + "\n\n".join(lines)


# ─── Command Handlers ─────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return

    user = update.effective_user.first_name or "there"
    stats = _db.get_stats()
    total = stats["total_posts"]
    g_stats = _graph.get_stats()

    welcome = (
        f"Hey {user}! Welcome to *InstaIntel*\n"
        f"{'=' * 30}\n\n"
        f"Your personal Instagram knowledge base.\n"
        f"Save posts, extract insights, search everything.\n\n"
    )

    if total > 0:
        top_topics = g_stats.get("top_topics", [])[:5]
        topics_str = ", ".join(f"#{t[0]}" for t in top_topics) if top_topics else "none yet"
        welcome += (
            f"*Your library:* {total} posts indexed\n"
            f"*Top topics:* {topics_str}\n\n"
        )
    else:
        welcome += "*Your library is empty!* Send an Instagram URL to get started.\n\n"

    welcome += (
        "*How to use:*\n"
        "1. Send any Instagram URL(s) to queue & analyze\n"
        "2. Ask anything about your saved content\n\n"
        "*Commands:*\n"
        "/stats  - Library overview\n"
        "/topics - Your top topics\n"
        "/recent - Last 5 saved posts\n"
        "/category - Browse by category\n"
        "/graph  - Knowledge graph visualization\n"
        "/queue  - Check processing queue\n"
        "/cost   - AI API usage & costs\n"
        "/flush  - Clear all saved data\n\n"
        "*Try asking:*\n"
        '_"What tips did I save about productivity?"_\n'
        '_"Summarize that fitness post"_\n\n'
        "Powered by Gemini + Claude"
    )

    await update.message.reply_text(welcome, parse_mode="Markdown")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    db_stats = _db.get_stats()
    v_stats = _vectors.get_stats()
    g_stats = _graph.get_stats()
    today_count = len(_db.get_today_posts())
    queue_count = len(_url_queue)

    text = (
        f"*InstaIntel Stats*\n"
        f"{'=' * 25}\n\n"
        f"Total posts: *{db_stats['total_posts']}*\n"
        f"Added today: *{today_count}*\n"
        f"Queue pending: *{queue_count}*\n\n"
    )

    type_icons = {"image": "IMG", "carousel": "SLIDES", "reel": "REEL"}
    for mtype, count in db_stats.get("by_type", {}).items():
        text += f"  [{type_icons.get(mtype, 'OTHER')}] {count}\n"

    text += (
        f"\n*Search index:*\n"
        f"  Posts: {v_stats['posts_indexed']} | Slides: {v_stats['slides_indexed']}\n"
        f"\n*Knowledge graph:*\n"
        f"  Nodes: {g_stats['total_nodes']} | Edges: {g_stats['total_edges']}\n"
    )

    cats = db_stats.get("by_category", {})
    if cats:
        text += "\n*Categories:*\n"
        for cat, count in list(cats.items())[:8]:
            text += f"  {cat}: {count}\n"

    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_topics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    topics = _graph.get_top_topics(15)
    if not topics:
        await update.message.reply_text("No topics yet. Send some Instagram URLs!")
        return

    lines = [f"  #{topic} ({degree})" for topic, degree in topics]
    await update.message.reply_text(
        "*Top Topics*\n\n" + "\n".join(lines), parse_mode="Markdown",
    )


async def cmd_recent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    posts = _db.get_all_posts()[:5]
    if not posts:
        await update.message.reply_text("No posts saved yet! Send an Instagram URL to start.")
        return

    lines = []
    for p in posts:
        author = p.get("author_username", "?")
        cat = p.get("category", "")
        summary = p.get("summary", "")[:100] or p.get("caption", "")[:100]
        url = p.get("url", "")
        lines.append(f"[{cat}] @{author}\n  {summary}\n  {url}")

    await update.message.reply_text(
        "*Recent Saves*\n\n" + "\n\n".join(lines), parse_mode="Markdown",
    )


async def cmd_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    category = " ".join(context.args) if context.args else ""
    if not category:
        stats = _db.get_stats()
        cats = stats.get("by_category", {})
        if cats:
            lines = [f"*{cat}* -- {count} posts" for cat, count in cats.items()]
            await update.message.reply_text(
                "Categories:\n\n" + "\n".join(lines) + "\n\nUsage: /category fitness",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text("No categories yet.")
        return

    posts = _db.get_posts_by_category(category)
    if not posts:
        await update.message.reply_text(f"No posts in: {category}")
        return

    lines = []
    for p in posts[:8]:
        author = p.get("author_username", "?")
        summary = p.get("summary", p.get("caption", ""))[:100]
        url = p.get("url", "")
        lines.append(f"@{author}: {summary}\n  {url}")

    await update.message.reply_text(
        f"*{category}* ({len(posts)} posts)\n\n" + "\n\n".join(lines),
        parse_mode="Markdown",
    )


# ─── Graph Visualization ───────────────────────────────────

async def cmd_graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    stats = _graph.get_stats()
    if stats["total_nodes"] == 0:
        await update.message.reply_text("No graph data yet. Send some Instagram URLs first!")
        return

    msg = await update.message.reply_text("Generating graph...")

    try:
        loop = asyncio.get_event_loop()
        html_path = await loop.run_in_executor(None, _graph.export_html)

        if html_path.exists():
            await update.message.reply_document(
                document=open(html_path, "rb"),
                filename="knowledge_graph.html",
                caption=(
                    f"*Knowledge Graph*\n"
                    f"Nodes: {stats['total_nodes']} | Edges: {stats['total_edges']}\n"
                    f"Open in browser to explore."
                ),
                parse_mode="Markdown",
            )
            await msg.delete()
        else:
            await msg.edit_text("Failed to generate graph.")
    except Exception as e:
        log.error("Graph export failed: %s", e)
        await msg.edit_text(f"Error: {e}")


# ─── Flush / Reset ────────────────────────────────────────────

async def cmd_flush(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    # Check for confirmation flag
    if not context.args or context.args[0] != "confirm":
        stats = _db.get_stats()
        await update.message.reply_text(
            f"This will permanently delete:\n"
            f"  - {stats['total_posts']} posts\n"
            f"  - All downloaded media files\n"
            f"  - Search index & knowledge graph\n\n"
            f"Type /flush confirm to proceed.",
        )
        return

    msg = await update.message.reply_text("Flushing all data...")

    try:
        # Clear database
        _db.flush_all()

        # Clear vector store
        _vectors.reset()

        # Clear knowledge graph
        _graph.G.clear()
        _graph.save()

        # Delete media files
        media_dir = Config.MEDIA_DIR
        if media_dir.exists():
            shutil.rmtree(media_dir)
            media_dir.mkdir(parents=True, exist_ok=True)

        await msg.edit_text(
            "All data has been flushed.\n"
            "Send an Instagram URL to start fresh!"
        )
        log.info("Full data flush completed by user %s", update.effective_user.id)

    except Exception as e:
        log.error("Flush failed: %s", e)
        await msg.edit_text(f"Flush error: {e}")


# ─── Cost Tracking ──────────────────────────────────────────

async def cmd_cost(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    vision_cost = _pipeline.vision.get_cost_summary()
    video_cost = _pipeline.gemini_video.get_cost_summary()
    entity_cost = _pipeline.extractor.get_cost_summary()

    # RAG chat costs (Claude Sonnet: $3/1M input, $15/1M output)
    rag_input_cost = (_rag_input_tokens / 1_000_000) * 3.0
    rag_output_cost = (_rag_output_tokens / 1_000_000) * 15.0
    rag_total = rag_input_cost + rag_output_cost

    total_cost = (
        vision_cost["estimated_cost_usd"]
        + video_cost["estimated_cost_usd"]
        + entity_cost["estimated_cost_usd"]
        + rag_total
    )

    text = (
        f"*AI Cost Tracker*\n"
        f"{'=' * 25}\n"
        f"_{Config.ANTHROPIC_CHAT_MODEL} + {Config.GEMINI_VIDEO_MODEL}_\n\n"

        f"*Gemini Vision (OCR)*\n"
        f"  Images: {vision_cost['total_images']}\n"
        f"  Tokens: {vision_cost['total_tokens']:,}\n"
        f"  Cost: ${vision_cost['estimated_cost_usd']:.4f}\n\n"

        f"*Gemini Video (Reels)*\n"
        f"  Videos: {video_cost['total_videos']}\n"
        f"  Tokens: {video_cost['total_tokens']:,}\n"
        f"  Cost: ${video_cost['estimated_cost_usd']:.4f}\n\n"

        f"*Claude Entity Extraction*\n"
        f"  Calls: {entity_cost['calls']}\n"
        f"  Tokens: {entity_cost['input_tokens'] + entity_cost['output_tokens']:,}\n"
        f"  Cost: ${entity_cost['estimated_cost_usd']:.4f}\n\n"

        f"*Claude RAG Chat*\n"
        f"  Queries: {_rag_calls}\n"
        f"  Tokens: {_rag_input_tokens + _rag_output_tokens:,}\n"
        f"  Cost: ${rag_total:.4f}\n\n"

        f"{'─' * 25}\n"
        f"*Total: ${total_cost:.4f}*\n"
        f"_{('This session only — resets on restart')}_"
    )

    await update.message.reply_text(text, parse_mode="Markdown")


# ─── Instagram URL Handler ────────────────────────────────────

async def handle_instagram_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    text = update.message.text.strip()
    urls = _INSTAGRAM_URL_RE.findall(text)
    if not urls:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    queued_count = 0
    skipped_dup = 0

    queued_urls = {item[0] for item in list(_url_queue)}

    for url in urls:
        m = _SHORTCODE_RE.search(url)
        if not m:
            continue
        shortcode = m.group(1)

        # Skip already processed
        if _db.is_url_processed(shortcode) or _db.get_post(shortcode):
            skipped_dup += 1
            continue

        # Skip if already in queue
        if url in queued_urls:
            continue

        _url_queue.append((url, chat_id, user_id))
        queued_urls.add(url)
        queued_count += 1

    # Build reply
    if queued_count == 0:
        if skipped_dup > 0:
            await update.message.reply_text(
                f"All {skipped_dup} URL(s) already saved. Ask me about them!"
            )
        return

    parts = [f"Added {queued_count} URL(s) to queue."]
    if skipped_dup:
        parts.append(f"({skipped_dup} already saved, skipped.)")

    await update.message.reply_text(" ".join(parts))
    log.info("Queued %d URLs from user %s. Queue size: %d", queued_count, user_id, len(_url_queue))


async def _send_images_album(update: Update, shortcode: str):
    """Send downloaded images as a Telegram photo album."""
    if not shortcode:
        return

    post_dir = Config.MEDIA_DIR / shortcode
    if not post_dir.exists():
        return

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(sorted(post_dir.glob(ext)))

    if not image_files:
        return

    # Telegram allows max 10 photos per album
    image_files = image_files[:10]

    try:
        if len(image_files) == 1:
            await update.message.reply_photo(photo=open(image_files[0], "rb"))
        else:
            media_group = [InputMediaPhoto(media=open(f, "rb")) for f in image_files]
            await update.message.reply_media_group(media=media_group)
    except Exception as e:
        log.warning("Failed to send images album: %s", e)


# ─── Queue Processor ─────────────────────────────────────────

async def _process_queue(context: ContextTypes.DEFAULT_TYPE):
    """Job: pop one URL from queue and process it silently. Runs every 60s."""
    if not _url_queue:
        return

    url, chat_id, user_id = _url_queue.popleft()
    log.info("[queue] Processing: %s (remaining: %d)", url, len(_url_queue))

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _pipeline.process_url, url)

        if result["status"] in ("duplicate", "processed"):
            _processing_stats["processed"] += 1
        else:
            _processing_stats["failed"] += 1
            _processing_stats["errors"].append({
                "url": url,
                "reason": result.get("error", "Unknown error"),
            })
    except Exception as e:
        log.error("[queue] Processing failed for %s: %s", url, e)
        _processing_stats["failed"] += 1
        _processing_stats["errors"].append({"url": url, "reason": str(e)})

    # Keep errors list bounded
    if len(_processing_stats["errors"]) > 20:
        _processing_stats["errors"] = _processing_stats["errors"][-20:]


async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show URL processing queue status."""
    if not _check_authorized(update.effective_user.id):
        return

    q_len = len(_url_queue)
    if q_len == 0:
        await update.message.reply_text("Queue is empty. Send Instagram URLs to add them!")
        return

    lines = [f"*Queue Status* — {q_len} URL(s) pending\n"]
    for i, (url, _, _) in enumerate(list(_url_queue)):
        sc = _SHORTCODE_RE.search(url)
        label = sc.group(1) if sc else url[-20:]
        lines.append(f"{i+1}. {label}")
        if i >= 9:
            lines.append(f"... and {q_len - 10} more")
            break

    lines.append(f"\nETA: ~{q_len} min (1 per minute)")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ─── Main Message Handler (Claude RAG) ────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _check_authorized(update.effective_user.id):
        return

    query = update.message.text.strip()
    if not query:
        return

    if _INSTAGRAM_URL_RE.search(query):
        return

    await update.message.chat.send_action("typing")

    from query import hybrid_search
    results = hybrid_search(query, _db, _vectors, _graph, n_results=8)

    if _claude:
        context_text = _build_context_from_results(results)
        answer = _ask_claude(query, context_text)
    else:
        answer = _fallback_response(results)

    try:
        if len(answer) <= 4000:
            await update.message.reply_text(
                answer, parse_mode="Markdown", disable_web_page_preview=True
            )
        else:
            chunks = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk, disable_web_page_preview=True)
    except Exception:
        await update.message.reply_text(answer, disable_web_page_preview=True)


# ─── Daily Digest (midnight IST = 18:30 UTC) ─────────────────

async def _daily_digest(context: ContextTypes.DEFAULT_TYPE):
    """Send end-of-day summary to all authorized users."""
    today_posts = _db.get_today_posts()
    total_stats = _db.get_stats()
    g_stats = _graph.get_stats()

    # Capture and reset processing stats
    processed = _processing_stats["processed"]
    failed = _processing_stats["failed"]
    errors_snapshot = list(_processing_stats["errors"])
    queue_remaining = len(_url_queue)

    _processing_stats["processed"] = 0
    _processing_stats["failed"] = 0
    _processing_stats["errors"] = []

    # Stats line
    total_attempts = processed + failed
    success_rate = int((processed / total_attempts) * 100) if total_attempts > 0 else 0
    stats_line = f"Processed: {processed} | Failed: {failed} | Success: {success_rate}%"
    if queue_remaining:
        stats_line += f" | Queue: {queue_remaining} pending"

    if not today_posts:
        text = (
            "*Daily Digest*\n"
            f"{'=' * 25}\n\n"
            "No new posts saved today.\n\n"
            f"*Processing:* {stats_line}\n\n"
            f"*Library total:* {total_stats['total_posts']} posts\n"
            f"*Graph:* {g_stats['total_nodes']} nodes\n\n"
            "Send some Instagram URLs tomorrow!"
        )
    else:
        # Group by category
        cats = {}
        for p in today_posts:
            cat = p.get("category", "general")
            cats.setdefault(cat, []).append(p)

        text = (
            "*Daily Digest*\n"
            f"{'=' * 25}\n\n"
            f"*Today:* {len(today_posts)} new posts saved\n"
            f"*Processing:* {stats_line}\n"
            f"*Library total:* {total_stats['total_posts']} posts\n\n"
        )

        for cat, posts in cats.items():
            text += f"*{cat}* ({len(posts)}):\n"
            for p in posts[:5]:
                author = p.get("author_username", "?")
                summary = p.get("summary", "")[:80]
                text += f"  @{author}: {summary}\n"
            text += "\n"

        # Top topics
        top_topics = g_stats.get("top_topics", [])[:5]
        if top_topics:
            topics_str = ", ".join(f"#{t[0]}" for t in top_topics)
            text += f"*Trending topics:* {topics_str}\n"

    # Append error details if any
    if errors_snapshot:
        text += f"\n*Errors ({len(errors_snapshot)}):*\n"
        for err in errors_snapshot[-10:]:
            short_url = err["url"].split("/")[-1][:20]
            reason = err["reason"][:80]
            text += f"  {short_url}: {reason}\n"

    # Send to all authorized users
    for user_id in Config.TELEGRAM_ALLOWED_USERS:
        try:
            await context.bot.send_message(
                chat_id=user_id, text=text, parse_mode="Markdown",
            )
        except Exception as e:
            log.error("Failed to send digest to %s: %s", user_id, e)

    # Clean up downloaded media files — already sent as albums
    media_dir = Config.MEDIA_DIR
    if media_dir.exists():
        freed = 0
        for item in media_dir.iterdir():
            if item.is_dir():
                freed += sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                shutil.rmtree(item)
        if freed:
            log.info("Nightly cleanup: freed %.1f MB of media", freed / (1024 * 1024))

    log.info("Daily digest sent: %d posts today, %d processed, %d failed", len(today_posts), processed, failed)


# ─── Bot Startup ──────────────────────────────────────────────

def start_bot():
    """Start the Telegram bot (blocking)."""
    global _db, _vectors, _graph, _claude, _pipeline

    if not Config.TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set!")
        return

    _db = Database()
    _vectors = VectorStore()
    _graph = KnowledgeGraph()

    downloader = ContentDownloader()
    vision = VisionProcessor()
    gemini_video = GeminiVideoProcessor()
    extractor = EntityExtractor()

    _pipeline = Pipeline(
        downloader=downloader, vision=vision, gemini_video=gemini_video,
        extractor=extractor, db=_db, vectors=_vectors, graph=_graph,
    )

    if Config.ANTHROPIC_API_KEY:
        _claude = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        log.info("Claude RAG chat enabled (model: %s)", Config.ANTHROPIC_CHAT_MODEL)
    else:
        log.warning("ANTHROPIC_API_KEY not set -- raw search results only")

    app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("topics", cmd_topics))
    app.add_handler(CommandHandler("recent", cmd_recent))
    app.add_handler(CommandHandler("category", cmd_category))
    app.add_handler(CommandHandler("graph", cmd_graph))
    app.add_handler(CommandHandler("flush", cmd_flush))
    app.add_handler(CommandHandler("cost", cmd_cost))
    app.add_handler(CommandHandler("queue", cmd_queue))

    # URL handler BEFORE catch-all text handler
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.Regex(_INSTAGRAM_URL_RE),
        handle_instagram_url,
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Daily digest at midnight IST (18:30 UTC)
    job_queue = app.job_queue
    job_queue.run_daily(
        _daily_digest,
        time=dt_time(hour=18, minute=30, tzinfo=timezone.utc),
        name="daily_digest",
    )
    log.info("Daily digest scheduled at 00:00 IST (18:30 UTC)")

    # Process queue: one URL every 60 seconds
    job_queue.run_repeating(
        _process_queue,
        interval=60,
        first=10,
        name="process_queue",
    )
    log.info("URL processing queue started (interval: 60s)")

    log.info(
        "Telegram bot started | Users: %s | Claude: %s",
        Config.TELEGRAM_ALLOWED_USERS,
        "enabled" if _claude else "disabled",
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    start_bot()
