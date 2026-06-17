"""Graph visualization — exports DOT files and optionally renders PNG via Graphviz."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .ir import Graph, Node

# ---------------------------------------------------------------------------
# DOT helpers
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _const_preview(value: np.ndarray) -> str:  # type: ignore[type-arg]
    if value.ndim == 0:
        return f"scalar={value.item()}"
    preview = repr(value.tolist())
    if len(preview) > 80:
        preview = preview[:77] + "..."
    return f"shape={tuple(value.shape)}\\n{preview}"


def _node_label(node: Node) -> str:
    parts = [f"op={node.op}"]
    if node.name:
        parts.append(f"name={node.name}")
    if node.op == "const" and isinstance(node.value, np.ndarray):
        parts.append(_const_preview(node.value))
    return "\\n".join(parts)


def _node_shape(node: Node) -> str:
    return {
        "input": "ellipse",
        "const": "box",
        "matmul": "doublecircle",
        "relu": "diamond",
        "fused_mma_bias_relu": "octagon",
    }.get(node.op, "circle")


def _node_style(
    node: Node,
    root: Node,
    reachable: set[Node] | None,
) -> tuple[str, str]:
    """Return (fillcolor, style) for a node."""
    if node is root:
        return "palegreen", "filled,bold"

    if reachable is not None and node not in reachable:
        return "mistyrose", "filled,dashed"

    return {
        "input": ("lightgray", "filled"),
        "const": ("lightgoldenrod1", "filled"),
        "matmul": ("lightblue", "filled"),
        "add": ("lightcyan", "filled"),
        "relu": ("plum1", "filled"),
        "fused_mma_bias_relu": ("orange", "filled,bold"),
    }.get(node.op, ("lightblue", "filled"))


# ---------------------------------------------------------------------------
# DOT writing
# ---------------------------------------------------------------------------

def _write_dot(
    nodes: list[Node],
    ids: dict[Node, str],
    root: Node,
    path: Path,
    title: str,
    reachable: set[Node] | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "digraph G {",
        '  rankdir="LR";',
        '  splines=true;',
        '  bgcolor="white";',
        f'  graph [fontsize=12, labelloc="t", label="{_escape(title)}"];',
        '  node [fontname="Consolas", fontsize=10, style="filled"];',
        '  edge [fontname="Consolas", fontsize=9, color="#555555"];',
    ]

    for n in nodes:
        nid = ids[n]
        label = _escape(_node_label(n))
        shape = _node_shape(n)
        fillcolor, style = _node_style(n, root, reachable)
        lines.append(
            f'  {nid} [label="{label}", shape={shape}, '
            f'fillcolor="{fillcolor}", style="{style}"];'
        )

    for n in nodes:
        src = ids[n]
        for i, ch in enumerate(n.inputs):
            if ch in ids:
                dst = ids[ch]
                lines.append(f'  {dst} -> {src} [label="{i}"];')

    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_dot(dot_path: str, png_path: str | None = None) -> tuple[str, bool, str]:
    """Render a DOT file to PNG via Graphviz ``dot``.

    Returns:
        (png_path, success, message)
    """
    dot_path = str(dot_path)
    if png_path is None:
        png_path = str(Path(dot_path).with_suffix(".png"))

    try:
        subprocess.run(
            ["dot", "-Tpng", dot_path, "-o", str(png_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return png_path, True, ""
    except FileNotFoundError:
        return png_path, False, "graphviz 'dot' not found — install Graphviz and add it to PATH"
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or str(e)).strip()
        return png_path, False, msg


def export_graph_dot(
    graph: Graph,
    root: Node,
    path: str,
    title: str = "Graph",
) -> str:
    """Export the *full* graph (``graph.nodes``) to DOT.

    Nodes not reachable from *root* are styled as dashed pink to highlight
    dead-code candidates.  *root* is highlighted in green.
    """
    nodes = list(graph.nodes)
    ids = {n: f"n{i}" for i, n in enumerate(nodes)}
    reachable = graph.reachable_from(root)
    return _write_dot(nodes, ids, root, Path(path), title, reachable=reachable)


def export_reachable_dot(
    root: Node,
    path: str,
    title: str = "Graph",
) -> str:
    """Export only the sub-graph reachable from *root* to DOT."""
    nodes: list[Node] = []
    seen: set[Node] = set()

    def visit(n: Node) -> None:
        if n in seen:
            return
        seen.add(n)
        for ch in n.inputs:
            visit(ch)
        nodes.append(n)

    visit(root)
    ids = {n: f"n{i}" for i, n in enumerate(nodes)}
    return _write_dot(nodes, ids, root, Path(path), title)


def dump_text_graph(root: Node) -> str:
    """Return an ASCII summary of the graph reachable from *root*."""
    order: list[Node] = []
    seen: set[Node] = set()

    def visit(n: Node) -> None:
        if n in seen:
            return
        seen.add(n)
        for ch in n.inputs:
            visit(ch)
        order.append(n)

    visit(root)

    lines = ["Graph summary:"]
    for i, n in enumerate(order):
        ins = ", ".join(ch.name or ch.op for ch in n.inputs) if n.inputs else "-"
        name = n.name or "-"
        lines.append(f"{i:02d}. op={n.op:<20} name={name:<12} inputs=[{ins}]")
    return "\n".join(lines)
