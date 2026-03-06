"""Knowledge Graph — NetworkX entity-relationship graph.

Inspired by Rowboat's knowledge graph pattern:
  post ↔ topic ↔ person ↔ brand ↔ product

Node types: post, topic, person, brand, product, location, category
Edge types: has_topic, mentions_person, mentions_brand, has_product,
            at_location, in_category, related_to
"""

import json
from pathlib import Path
from collections import Counter

import networkx as nx

from config import Config
from core.entity_extractor import ExtractedEntities
from utils.logger import log


class KnowledgeGraph:
    """Manages the knowledge graph for content intelligence."""

    def __init__(self):
        Config.ensure_dirs()
        self.graph_path = Config.GRAPH_PATH
        self.G = self._load_or_create()
        log.info(
            "Knowledge graph loaded: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    def _load_or_create(self) -> nx.Graph:
        """Load existing graph or create new one."""
        if self.graph_path.exists():
            try:
                data = json.loads(self.graph_path.read_text())
                G = nx.node_link_graph(data)
                return G
            except Exception as e:
                log.warning("Failed to load graph, creating new: %s", e)
        return nx.Graph()

    def save(self):
        """Persist graph to disk."""
        data = nx.node_link_data(self.G)
        self.graph_path.write_text(json.dumps(data, indent=2, default=str))
        log.debug("Graph saved: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges())

    # ─── Add content ──────────────────────────────────────────────────

    def add_post(
        self,
        media_id: str,
        shortcode: str,
        author: str,
        entities: ExtractedEntities,
        url: str = "",
    ):
        """Add a post and all its entities to the graph."""

        # Post node
        post_id = f"post:{media_id}"
        self.G.add_node(
            post_id,
            node_type="post",
            shortcode=shortcode,
            author=author,
            url=url,
            summary=entities.summary,
            category=entities.category,
        )

        # Category node + edge
        cat_id = f"category:{entities.category}"
        self.G.add_node(cat_id, node_type="category", name=entities.category)
        self.G.add_edge(post_id, cat_id, edge_type="in_category")

        # Topics
        for topic in entities.topics:
            topic_clean = topic.lower().strip()
            if not topic_clean:
                continue
            t_id = f"topic:{topic_clean}"
            self.G.add_node(t_id, node_type="topic", name=topic_clean)
            self.G.add_edge(post_id, t_id, edge_type="has_topic")

        # People
        for person in entities.people:
            person_clean = person.lower().strip()
            if not person_clean:
                continue
            p_id = f"person:{person_clean}"
            self.G.add_node(p_id, node_type="person", name=person)
            self.G.add_edge(post_id, p_id, edge_type="mentions_person")

        # Brands
        for brand in entities.brands:
            brand_clean = brand.lower().strip()
            if not brand_clean:
                continue
            b_id = f"brand:{brand_clean}"
            self.G.add_node(b_id, node_type="brand", name=brand)
            self.G.add_edge(post_id, b_id, edge_type="mentions_brand")

        # Products
        for product in entities.products:
            product_clean = product.lower().strip()
            if not product_clean:
                continue
            pr_id = f"product:{product_clean}"
            self.G.add_node(pr_id, node_type="product", name=product)
            self.G.add_edge(post_id, pr_id, edge_type="has_product")

        # Locations
        for loc in entities.locations:
            loc_clean = loc.lower().strip()
            if not loc_clean:
                continue
            l_id = f"location:{loc_clean}"
            self.G.add_node(l_id, node_type="location", name=loc)
            self.G.add_edge(post_id, l_id, edge_type="at_location")

        # Author node
        if author:
            a_id = f"person:@{author.lower()}"
            self.G.add_node(a_id, node_type="person", name=f"@{author}")
            self.G.add_edge(post_id, a_id, edge_type="authored_by")

        # Cross-link co-occurring topics
        topic_ids = [f"topic:{t.lower().strip()}" for t in entities.topics if t.strip()]
        for i, t1 in enumerate(topic_ids):
            for t2 in topic_ids[i + 1:]:
                if self.G.has_edge(t1, t2):
                    self.G[t1][t2]["weight"] = self.G[t1][t2].get("weight", 1) + 1
                else:
                    self.G.add_edge(t1, t2, edge_type="related_to", weight=1)

        self.save()
        log.info("Added post %s to graph with %d entities", shortcode, sum([
            len(entities.topics), len(entities.people), len(entities.brands),
            len(entities.products), len(entities.locations),
        ]))

    # ─── Query ────────────────────────────────────────────────────────

    def find_related_posts(self, query_terms: list[str], max_results: int = 10) -> list[dict]:
        """Find posts related to query terms via graph traversal.

        Strategy: find matching entity nodes → traverse to connected posts.
        """
        matching_posts = {}

        for term in query_terms:
            term_lower = term.lower().strip()

            # Search all non-post nodes for matching names
            for node, data in self.G.nodes(data=True):
                if data.get("node_type") == "post":
                    continue
                name = data.get("name", "").lower()
                if term_lower in name or name in term_lower:
                    # Get all posts connected to this entity
                    for neighbor in self.G.neighbors(node):
                        n_data = self.G.nodes[neighbor]
                        if n_data.get("node_type") == "post":
                            if neighbor not in matching_posts:
                                matching_posts[neighbor] = {
                                    "post_id": neighbor,
                                    "shortcode": n_data.get("shortcode", ""),
                                    "author": n_data.get("author", ""),
                                    "summary": n_data.get("summary", ""),
                                    "category": n_data.get("category", ""),
                                    "url": n_data.get("url", ""),
                                    "match_score": 0,
                                    "matched_via": [],
                                }
                            matching_posts[neighbor]["match_score"] += 1
                            matching_posts[neighbor]["matched_via"].append(name)

        # Sort by score (most connected matches first)
        results = sorted(
            matching_posts.values(),
            key=lambda x: x["match_score"],
            reverse=True,
        )[:max_results]

        return results

    def get_neighborhood(self, node_id: str, depth: int = 2) -> dict:
        """Get the subgraph around a node for visualization."""
        if node_id not in self.G:
            return {"nodes": [], "edges": []}

        # BFS to depth
        visited = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                for neighbor in self.G.neighbors(n):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # Build subgraph data
        nodes = []
        for n in visited:
            data = dict(self.G.nodes[n])
            data["id"] = n
            nodes.append(data)

        edges = []
        for u, v, data in self.G.edges(data=True):
            if u in visited and v in visited:
                edges.append({"source": u, "target": v, **data})

        return {"nodes": nodes, "edges": edges}

    def get_top_topics(self, n: int = 20) -> list[tuple[str, int]]:
        """Get most connected topics."""
        topic_degrees = []
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "topic":
                degree = self.G.degree(node)
                topic_degrees.append((data.get("name", node), degree))

        return sorted(topic_degrees, key=lambda x: x[1], reverse=True)[:n]

    def get_stats(self) -> dict:
        """Graph statistics."""
        type_counts = Counter()
        for _, data in self.G.nodes(data=True):
            type_counts[data.get("node_type", "unknown")] += 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "node_types": dict(type_counts),
            "top_topics": self.get_top_topics(10),
        }

    # ─── Visualization ────────────────────────────────────────────────

    def export_html(self, output_path: Path | None = None) -> Path:
        """Export interactive graph visualization as HTML using pyvis."""
        try:
            from pyvis.network import Network

            net = Network(
                height="800px",
                width="100%",
                bgcolor="#0a0a0f",
                font_color="#e2e8f0",
                directed=False,
            )

            # Color map by node type
            colors = {
                "post": "#8b5cf6",
                "topic": "#22c55e",
                "person": "#3b82f6",
                "brand": "#f59e0b",
                "product": "#ec4899",
                "location": "#06b6d4",
                "category": "#ef4444",
            }

            for node, data in self.G.nodes(data=True):
                node_type = data.get("node_type", "unknown")
                label = data.get("name", data.get("shortcode", node))
                color = colors.get(node_type, "#64748b")
                size = 15 if node_type == "post" else 10
                net.add_node(node, label=label, color=color, size=size, title=f"{node_type}: {label}")

            for u, v, data in self.G.edges(data=True):
                weight = data.get("weight", 1)
                net.add_edge(u, v, width=min(weight, 5))

            out = output_path or (Config.DATA_DIR / "graph.html")
            net.save_graph(str(out))
            log.info("Graph visualization exported to %s", out)
            return out

        except ImportError:
            log.warning("pyvis not installed. Run: pip install pyvis")
            return Config.DATA_DIR / "graph.html"
