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
        """Export interactive graph visualization as a self-contained vis.js HTML file."""
        out = output_path or (Config.DATA_DIR / "graph.html")

        TYPE_COLORS = {
            "post": "#8b5cf6", "topic": "#22c55e", "person": "#3b82f6",
            "brand": "#f59e0b", "product": "#ec4899", "location": "#06b6d4",
            "category": "#ef4444",
        }
        TYPE_SHAPES = {
            "post": "triangle", "topic": "dot", "person": "square",
            "brand": "diamond", "product": "star", "location": "ellipse",
            "category": "hexagon",
        }

        vis_nodes = []
        for node, data in self.G.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            raw_label = data.get("name", data.get("shortcode", node))
            label = (raw_label[:25] + "\u2026") if len(raw_label) > 25 else raw_label
            vis_nodes.append({
                "id": node, "label": label,
                "title": f"{node_type}: {raw_label}",
                "color": TYPE_COLORS.get(node_type, "#64748b"),
                "shape": TYPE_SHAPES.get(node_type, "dot"),
                "size": 20 if node_type == "post" else 12,
                "font": {"size": 11, "color": "#e6edf3"},
            })

        vis_edges = []
        for u, v, data in self.G.edges(data=True):
            weight = data.get("weight", 1)
            vis_edges.append({
                "from": u, "to": v, "width": min(weight, 5),
                "color": {"color": "#334155", "highlight": "#94a3b8"},
            })

        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)

        legend_items = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:2px"></span>'
            f'<span style="font-size:11px;color:#94a3b8">{ntype}</span></div>'
            for ntype, color in TYPE_COLORS.items()
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>InstaIntel Knowledge Graph</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/dist/vis-network.min.css">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,sans-serif;overflow:hidden}}
#graph{{width:100vw;height:100vh}}
#controls{{position:absolute;top:12px;left:12px;z-index:10;display:flex;gap:8px;align-items:center}}
#search{{background:#161b22;border:1px solid #30363d;color:#e6edf3;padding:6px 10px;border-radius:6px;font-size:13px;width:200px;outline:none}}
#search::placeholder{{color:#6e7681}}
.btn{{background:#21262d;border:1px solid #30363d;color:#e6edf3;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:13px}}
.btn:hover{{background:#30363d}}
#legend{{position:absolute;top:12px;right:12px;z-index:10;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px}}
#legend h4{{font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
#node-info{{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 16px;font-size:13px;color:#e6edf3;max-width:500px;text-align:center;display:none;z-index:10}}
</style>
</head>
<body>
<div id="controls">
  <input id="search" type="text" placeholder="Search nodes\u2026"/>
  <button class="btn" id="search-btn">Find</button>
  <button class="btn" id="reset-btn">Reset view</button>
</div>
<div id="legend"><h4>Legend</h4>{legend_items}</div>
<div id="graph"></div>
<div id="node-info"></div>
<script>
const nodesData={nodes_json};
const edgesData={edges_json};
const nodes=new vis.DataSet(nodesData);
const edges=new vis.DataSet(edgesData);
const options={{
  physics:{{solver:"barnesHut",barnesHut:{{gravitationalConstant:-8000,centralGravity:0.3,springLength:150,springConstant:0.04,damping:0.09}},stabilization:{{iterations:200}}}},
  interaction:{{hover:true,tooltipDelay:200,hideEdgesOnDrag:true}},
  edges:{{smooth:{{type:"continuous"}},color:{{color:"#334155",highlight:"#94a3b8"}}}},
  nodes:{{borderWidth:0,chosen:true}}
}};
const container=document.getElementById("graph");
const network=new vis.Network(container,{{nodes,edges}},options);
network.on("click",function(params){{
  const info=document.getElementById("node-info");
  if(params.nodes.length===0){{
    nodes.update(nodesData.map(n=>({{id:n.id,opacity:1.0}})));
    info.style.display="none";
    return;
  }}
  const nodeId=params.nodes[0];
  const nd=nodes.get(nodeId);
  info.textContent=nd.title||nd.label;
  info.style.display="block";
  const connected=new Set(network.getConnectedNodes(nodeId));
  connected.add(nodeId);
  nodes.update(nodesData.map(n=>({{id:n.id,opacity:connected.has(n.id)?1.0:0.1}})));
}});
document.getElementById("search-btn").addEventListener("click",()=>{{
  const q=document.getElementById("search").value.trim().toLowerCase();
  if(!q)return;
  const match=nodesData.find(n=>n.label.toLowerCase().includes(q)||n.title.toLowerCase().includes(q));
  if(match){{network.focus(match.id,{{scale:1.5,animation:true}});network.selectNodes([match.id]);}}
}});
document.getElementById("search").addEventListener("keydown",e=>{{
  if(e.key==="Enter")document.getElementById("search-btn").click();
}});
document.getElementById("reset-btn").addEventListener("click",()=>{{
  network.fit({{animation:true}});
  nodes.update(nodesData.map(n=>({{id:n.id,opacity:1.0}})));
  document.getElementById("node-info").style.display="none";
}});
</script>
</body>
</html>"""

        out.write_text(html, encoding="utf-8")
        log.info("Graph exported to %s (%d nodes, %d edges)", out, len(vis_nodes), len(vis_edges))
        return out
