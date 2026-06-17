"""Unit tests for the visualization module."""

import tempfile
from pathlib import Path

from compiler.ir import Graph
from compiler.viz import (
    dump_text_graph,
    export_graph_dot,
    export_reachable_dot,
    render_dot,
)


class TestDumpTextGraph:
    def test_simple_graph(self):
        g = Graph()
        x = g.Input("x")
        w = g.Const([1.0, 2.0])
        out = g.Add(x, w)
        g.output = out
        text = dump_text_graph(out)
        assert "Graph summary" in text
        assert "input" in text
        assert "const" in text
        assert "add" in text

    def test_named_nodes_appear(self):
        g = Graph()
        x = g.Input("my_input")
        out = g.ReLU(x, name="activation")
        text = dump_text_graph(out)
        assert "my_input" in text
        assert "activation" in text


class TestExportDot:
    def test_export_graph_dot_writes_file(self):
        g = Graph()
        x = g.Input("x")
        out = g.ReLU(x)
        g.output = out
        with tempfile.TemporaryDirectory() as tmp:
            path = export_graph_dot(g, out, str(Path(tmp) / "test.dot"))
            assert path.endswith(".dot")
            content = Path(path).read_text()
            assert "digraph G" in content
            assert "relu" in content.lower()

    def test_export_reachable_dot_omits_dead(self):
        g = Graph()
        x = g.Input("x")
        dead = g.Const([99])
        out = g.ReLU(x)
        g.output = out

        with tempfile.TemporaryDirectory() as tmp:
            path = export_reachable_dot(out, str(Path(tmp) / "test.dot"))
            content = Path(path).read_text()
            # Dead node should not appear.
            assert "99" not in content

    def test_export_full_graph_includes_dead(self):
        g = Graph()
        x = g.Input("x")
        dead = g.Const([99])
        out = g.ReLU(x)
        g.output = out

        with tempfile.TemporaryDirectory() as tmp:
            path = export_graph_dot(g, out, str(Path(tmp) / "test.dot"))
            content = Path(path).read_text()
            # Dead node should appear (dashed pink).
            assert "99" in content


class TestRenderDot:
    def test_missing_dot_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            dot_path = Path(tmp) / "test.dot"
            dot_path.write_text("digraph G {}")
            png, ok, msg = render_dot(str(dot_path))
            if not ok:
                assert "not found" in msg.lower() or "graphviz" in msg.lower()
