"""Query Interface — CLI for searching your saved Instagram content.

Combines: vector semantic search + graph traversal + full-text SQLite search.

Usage:
    python query.py "skincare tips"
    python query.py --category fitness
    python query.py --graph "show my knowledge graph"
    python query.py --stats
"""

import json
import sys

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from config import Config
from storage.database import Database
from storage.vector_store import VectorStore
from storage.knowledge_graph import KnowledgeGraph
from utils.logger import log

console = Console()


def hybrid_search(
    query: str,
    db: Database,
    vectors: VectorStore,
    graph: KnowledgeGraph,
    n_results: int = 10,
) -> list[dict]:
    """Combine vector search, graph traversal, and full-text search.

    Scoring:
    - Vector match:  base score from cosine similarity
    - Graph match:   bonus for graph-connected results
    - Text match:    bonus for exact text matches
    """
    results = {}

    # 1. Vector semantic search
    vector_hits = vectors.search(query, n_results=n_results)
    for hit in vector_hits:
        media_id = hit["id"]
        # Cosine distance → similarity (lower distance = better)
        similarity = max(0, 1 - hit.get("distance", 1))
        results[media_id] = {
            "media_id": media_id,
            "vector_score": similarity,
            "graph_score": 0,
            "text_score": 0,
            "sources": ["vector"],
        }

    # 2. Graph traversal
    query_terms = query.lower().split()
    graph_hits = graph.find_related_posts(query_terms, max_results=n_results)
    for hit in graph_hits:
        media_id = hit["post_id"].replace("post:", "")
        if media_id in results:
            results[media_id]["graph_score"] = hit["match_score"] * Config.SEARCH_GRAPH_WEIGHT
            results[media_id]["sources"].append("graph")
            results[media_id]["matched_via"] = hit.get("matched_via", [])
        else:
            results[media_id] = {
                "media_id": media_id,
                "vector_score": 0,
                "graph_score": hit["match_score"] * Config.SEARCH_GRAPH_WEIGHT,
                "text_score": 0,
                "sources": ["graph"],
                "matched_via": hit.get("matched_via", []),
            }

    # 3. Full-text search
    text_hits = db.search_posts(query)
    for hit in text_hits:
        media_id = hit["media_id"]
        if media_id in results:
            results[media_id]["text_score"] = Config.SEARCH_TEXT_WEIGHT
            results[media_id]["sources"].append("text")
        else:
            results[media_id] = {
                "media_id": media_id,
                "vector_score": 0,
                "graph_score": 0,
                "text_score": Config.SEARCH_TEXT_WEIGHT,
                "sources": ["text"],
            }

    # Combine scores and sort
    for r in results.values():
        r["total_score"] = r["vector_score"] + r["graph_score"] + r["text_score"]

    ranked = sorted(results.values(), key=lambda x: x["total_score"], reverse=True)

    # Enrich with DB metadata
    enriched = []
    for r in ranked[:n_results]:
        post = db.get_post(r["media_id"])
        if post:
            r["post"] = post
            enriched.append(r)

    return enriched


def format_result(r: dict, index: int) -> Panel:
    """Format a search result as a Rich panel."""
    post = r.get("post", {})
    score = r["total_score"]
    sources = ", ".join(r["sources"])

    title = f"#{index + 1} | @{post.get('author_username', '?')} | {post.get('media_type', '?')} | {post.get('category', '?')}"

    content_parts = []

    if post.get("summary"):
        content_parts.append(f"**Summary:** {post['summary']}")

    if post.get("caption"):
        cap = post["caption"][:200]
        if len(post.get("caption", "")) > 200:
            cap += "..."
        content_parts.append(f"\n**Caption:** {cap}")

    if post.get("ocr_text"):
        ocr = post["ocr_text"][:300]
        if len(post.get("ocr_text", "")) > 300:
            ocr += "..."
        content_parts.append(f"\n**Image Text:** {ocr}")

    # Show tips if present
    try:
        entities = json.loads(post.get("entities_json", "{}"))
        tips = entities.get("tips", [])
        if tips:
            tip_text = "\n".join(f"  • {t}" for t in tips[:3])
            content_parts.append(f"\n**Tips:**\n{tip_text}")
    except (json.JSONDecodeError, TypeError):
        pass

    if post.get("url"):
        content_parts.append(f"\n🔗 {post['url']}")

    content_parts.append(f"\n_Score: {score:.2f} | via: {sources}_")

    # Show graph match info
    if "matched_via" in r and r["matched_via"]:
        content_parts.append(f"_Graph links: {', '.join(r['matched_via'][:5])}_")

    content = "\n".join(content_parts)

    return Panel(
        Markdown(content),
        title=title,
        border_style="bright_magenta" if score > 0.5 else "dim",
    )


@click.command()
@click.argument("query", required=False, default=None)
@click.option("--category", "-c", help="Filter by category")
@click.option("--graph", "-g", is_flag=True, help="Export graph visualization")
@click.option("--stats", "-s", is_flag=True, help="Show pipeline stats")
@click.option("--top", "-n", default=5, help="Number of results")
def main(query, category, graph, stats, top):
    """Search your saved Instagram content."""

    Config.ensure_dirs()
    db = Database()
    vectors = VectorStore()
    kg = KnowledgeGraph()

    if stats:
        _show_stats(db, vectors, kg)
        return

    if graph:
        out = kg.export_html()
        console.print(f"\n✓ Graph exported to: [link file://{out}]{out}[/link]")
        console.print("  Open in browser to explore your knowledge graph.")
        return

    if category:
        posts = db.get_posts_by_category(category)
        console.print(f"\n[bold]Posts in category: {category}[/bold] ({len(posts)} found)\n")
        for i, post in enumerate(posts[:top]):
            console.print(format_result({"post": post, "total_score": 0, "sources": ["category"]}, i))
        return

    if not query:
        console.print("\n[bold]InstaIntel[/bold] — Query your saved Instagram content\n")
        console.print("Usage:")
        console.print('  python query.py "skincare routine"')
        console.print("  python query.py --category fitness")
        console.print("  python query.py --graph")
        console.print("  python query.py --stats")
        _show_stats(db, vectors, kg)
        return

    # Hybrid search
    console.print(f"\n[bold]Searching:[/bold] {query}\n")
    results = hybrid_search(query, db, vectors, kg, n_results=top)

    if not results:
        console.print("[dim]No results found. Try a different query or save more posts![/dim]")
        return

    for i, r in enumerate(results):
        console.print(format_result(r, i))
        console.print()


def _show_stats(db: Database, vectors: VectorStore, kg: KnowledgeGraph):
    """Display pipeline statistics."""
    db_stats = db.get_stats()
    v_stats = vectors.get_stats()
    g_stats = kg.get_stats()

    table = Table(title="InstaIntel Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Posts", str(db_stats["total_posts"]))
    for mtype, count in db_stats.get("by_type", {}).items():
        table.add_row(f"  {mtype}", str(count))
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Vector Index (posts)", str(v_stats["posts_indexed"]))
    table.add_row("Vector Index (slides)", str(v_stats["slides_indexed"]))
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Graph Nodes", str(g_stats["total_nodes"]))
    table.add_row("Graph Edges", str(g_stats["total_edges"]))

    if g_stats.get("top_topics"):
        table.add_row("─" * 20, "─" * 10)
        for topic, degree in g_stats["top_topics"][:5]:
            table.add_row(f"  #{topic}", f"{degree} connections")

    if db_stats.get("by_category"):
        table.add_row("─" * 20, "─" * 10)
        for cat, count in list(db_stats["by_category"].items())[:8]:
            table.add_row(f"  📁 {cat}", str(count))

    console.print()
    console.print(table)


if __name__ == "__main__":
    main()
