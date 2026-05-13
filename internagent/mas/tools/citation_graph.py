"""
Citation Graph for InternAgent

Provides a directed citation graph that stores paper metadata as node attributes
and directed citation edges (source → cited_paper).  Supports:
- JSON serialisation (nodes + edges)
- DOT export (via NetworkX / pydot)
- PNG rendering (requires graphviz system package or matplotlib fallback)
- LLM-extracted ``problem_and_background``, ``contributions``,
  ``methods``, ``challenges``, ``limitations_and_future_work`` fields
  on every node that has been deep-read.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    import subprocess
    subprocess.check_call(["pip", "install", "networkx"])
    import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node attribute keys that carry LLM-extracted analysis
# ---------------------------------------------------------------------------
_ANALYSIS_KEYS = (
    "problem_and_background",
    "contributions",
    "methods",
    "challenges",
    "limitations_and_future_work",
)


class CitationGraph:
    """
    Directed citation graph where each node represents a paper and each edge
    ``(A, B)`` means *paper A cites paper B*.

    Node attributes include:
    - ``title``         – paper title
    - ``authors``       – list of author names
    - ``year``          – publication year (int or None)
    - ``abstract``      – abstract text
    - ``doi``           – DOI string or None
    - ``url``           – canonical URL or None
    - ``pdf_url``       – open-access PDF URL or None
    - ``citations``     – total citation count (int or None)
    - ``source``        – data source (``semantic_scholar``, ``arxiv``, …)
    - ``paper_id``      – Semantic Scholar paper ID or None
    - ``depth``         – BFS depth at which this node was discovered
    - ``problem_and_background``    – LLM-extracted (populated on deep read)
    - ``contributions``             – LLM-extracted
    - ``methods``                   – LLM-extracted
    - ``challenges``                – LLM-extracted
    - ``limitations_and_future_work`` – LLM-extracted

    Edge attributes:
    - ``relation`` – always ``"cites"``
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        # Map title (lower-cased) → node key for duplicate detection
        self._title_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_key(self, paper: Dict[str, Any]) -> str:
        """Return a stable node key for a paper dict."""
        pid = paper.get("paper_id")
        if pid:
            return pid
        doi = paper.get("doi")
        if doi:
            return f"doi:{doi}"
        title = (paper.get("title") or "").lower().strip()
        return f"title:{title}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_paper(self, paper: Dict[str, Any], depth: int = 0) -> str:
        """
        Add a paper to the graph (idempotent: duplicates are ignored).

        Args:
            paper: Dictionary with at minimum a ``title`` key.  All standard
                   :class:`~internagent.mas.tools.literature_search.PaperMetadata`
                   dict keys are recognised.
            depth: BFS depth at which this paper was discovered.

        Returns:
            The node key used in the underlying :class:`networkx.DiGraph`.
        """
        key = self._node_key(paper)
        if not self._g.has_node(key):
            attrs: Dict[str, Any] = {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "abstract": paper.get("abstract", ""),
                "doi": paper.get("doi"),
                "url": paper.get("url"),
                "pdf_url": paper.get("pdf_url"),
                "citations": paper.get("citations"),
                "source": paper.get("source", "unknown"),
                "paper_id": paper.get("paper_id"),
                "depth": depth,
            }
            for ak in _ANALYSIS_KEYS:
                attrs[ak] = paper.get(ak)
            self._g.add_node(key, **attrs)

            title_lower = attrs["title"].lower().strip()
            if title_lower:
                self._title_index[title_lower] = key
        else:
            # Update depth if discovered closer to root
            existing_depth = self._g.nodes[key].get("depth", depth)
            if depth < existing_depth:
                self._g.nodes[key]["depth"] = depth
            # Backfill analysis fields if not yet set
            for ak in _ANALYSIS_KEYS:
                if self._g.nodes[key].get(ak) is None and paper.get(ak) is not None:
                    self._g.nodes[key][ak] = paper[ak]
        return key

    def add_citation(self, source_key: str, target_key: str) -> None:
        """
        Add a directed citation edge: ``source`` cites ``target``.

        Both nodes must already exist in the graph (call :meth:`add_paper`
        first).  Duplicate edges are silently ignored.

        Args:
            source_key: Node key of the citing paper.
            target_key: Node key of the cited paper.
        """
        if not self._g.has_node(source_key) or not self._g.has_node(target_key):
            logger.warning(
                f"[CitationGraph] Skipping edge: node not found "
                f"({source_key!r} -> {target_key!r})"
            )
            return
        if not self._g.has_edge(source_key, target_key):
            self._g.add_edge(source_key, target_key, relation="cites")

    def update_analysis(self, node_key: str, analysis: Dict[str, Any]) -> None:
        """
        Attach LLM-extracted analysis fields to an existing node.

        Args:
            node_key: The key of the node to update.
            analysis: Dict with any subset of the :data:`_ANALYSIS_KEYS`.
        """
        if not self._g.has_node(node_key):
            logger.warning(f"[CitationGraph] Node not found: {node_key!r}")
            return
        for ak in _ANALYSIS_KEYS:
            if ak in analysis and analysis[ak] is not None:
                self._g.nodes[node_key][ak] = analysis[ak]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        """
        Serialise the graph to a JSON-compatible dictionary.

        Returns a dict with two keys:
        - ``nodes`` – list of ``{id, **attributes}`` dicts
        - ``edges`` – list of ``{source, target, relation}`` dicts
        """
        nodes = []
        for nid, attrs in self._g.nodes(data=True):
            node_dict = {"id": nid}
            node_dict.update(attrs)
            nodes.append(node_dict)

        edges = []
        for src, tgt, edata in self._g.edges(data=True):
            edges.append({"source": src, "target": tgt, **edata})

        return {"nodes": nodes, "edges": edges}

    def save_json(self, path: str) -> None:
        """Write the graph to a JSON file at *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, ensure_ascii=False, indent=2)
        logger.info(f"[CitationGraph] JSON saved to {path}")

    def to_dot(self) -> str:
        """
        Export the graph as a Graphviz DOT string.

        Node labels show title (truncated to 40 chars), year, and a colour
        gradient based on BFS depth.
        """
        try:
            from networkx.drawing.nx_pydot import to_pydot
            pdot = to_pydot(self._g)
            return pdot.to_string()
        except Exception:
            # Fallback: hand-crafted DOT
            lines = ["digraph CitationGraph {", "  rankdir=LR;"]
            for nid, attrs in self._g.nodes(data=True):
                label = (attrs.get("title") or nid)[:40].replace('"', "'")
                year = attrs.get("year") or ""
                depth = attrs.get("depth", 0)
                # Light-blue shading gets darker with depth
                shade = max(0, 255 - depth * 40)
                color = f"#{shade:02x}{shade:02x}ff"
                lines.append(
                    f'  "{nid}" [label="{label}\\n{year}" '
                    f'style=filled fillcolor="{color}"];'
                )
            for src, tgt in self._g.edges():
                lines.append(f'  "{src}" -> "{tgt}";')
            lines.append("}")
            return "\n".join(lines)

    def save_dot(self, path: str) -> None:
        """Write the DOT representation to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_dot())
        logger.info(f"[CitationGraph] DOT saved to {path}")

    def save_png(self, path: str) -> bool:
        """
        Render the graph to a PNG image at *path*.

        Tries three backends in order:
        1. ``graphviz`` system binary (``dot``).
        2. ``matplotlib`` + NetworkX spring layout.
        3. Returns ``False`` if neither is available.

        Args:
            path: Destination PNG file path.

        Returns:
            ``True`` on success, ``False`` if rendering was not possible.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # --- try graphviz dot binary ---
        import shutil, subprocess as _sp
        dot_bin = shutil.which("dot")
        if dot_bin:
            dot_src = self.to_dot()
            dot_tmp = path + ".dot"
            try:
                with open(dot_tmp, "w", encoding="utf-8") as fh:
                    fh.write(dot_src)
                result = _sp.run(
                    [dot_bin, "-Tpng", dot_tmp, "-o", path],
                    capture_output=True,
                    timeout=60,
                )
                os.remove(dot_tmp)
                if result.returncode == 0:
                    logger.info(f"[CitationGraph] PNG saved via graphviz to {path}")
                    return True
                logger.warning(
                    f"[CitationGraph] graphviz failed: {result.stderr.decode()}"
                )
            except Exception as e:
                logger.warning(f"[CitationGraph] graphviz error: {e}")
            finally:
                if os.path.exists(dot_tmp):
                    os.remove(dot_tmp)

        # --- try matplotlib ---
        try:
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore

            fig, ax = plt.subplots(figsize=(16, 10))
            pos = nx.spring_layout(self._g, k=2.0, seed=42)
            labels = {
                n: (attrs.get("title") or n)[:30]
                for n, attrs in self._g.nodes(data=True)
            }
            depths = [self._g.nodes[n].get("depth", 0) for n in self._g.nodes()]
            nx.draw_networkx(
                self._g,
                pos=pos,
                labels=labels,
                node_color=depths,
                cmap=plt.cm.Blues,
                node_size=800,
                font_size=7,
                arrows=True,
                ax=ax,
            )
            ax.set_title("Citation Graph", fontsize=14)
            plt.tight_layout()
            plt.savefig(path, dpi=120)
            plt.close(fig)
            logger.info(f"[CitationGraph] PNG saved via matplotlib to {path}")
            return True
        except Exception as e:
            logger.warning(f"[CitationGraph] matplotlib rendering failed: {e}")

        logger.error("[CitationGraph] Could not render PNG: no suitable backend.")
        return False

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def node_keys(self) -> List[str]:
        """Return all node keys."""
        return list(self._g.nodes())

    def get_node(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the attribute dict for *key*, or ``None`` if absent."""
        if self._g.has_node(key):
            return dict(self._g.nodes[key])
        return None

    def successors(self, key: str) -> List[str]:
        """Return node keys that *key* cites (outgoing edges)."""
        return list(self._g.successors(key))

    def predecessors(self, key: str) -> List[str]:
        """Return node keys that cite *key* (incoming edges)."""
        return list(self._g.predecessors(key))

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def __contains__(self, key: str) -> bool:
        return self._g.has_node(key)
