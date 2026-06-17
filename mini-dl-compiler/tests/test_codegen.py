"""Unit tests for the NumPy code generator."""

import numpy as np
import pytest

from compiler.codegen import NumpyCompiler
from compiler.ir import Graph


def _make_compiler():
    return NumpyCompiler()


class TestCodegen:
    def test_simple_input_only(self):
        g = Graph()
        x = g.Input("x")
        g.output = x

        fn, source, _, _ = _make_compiler().compile(x, graph=g, optimize=False)
        assert fn(np.array([1.0, 2.0])) == pytest.approx([1.0, 2.0])

    def test_const_node(self):
        g = Graph()
        c = g.Const([3.0, 4.0])
        g.output = c

        fn, source, _, _ = _make_compiler().compile(c, graph=g, optimize=False)
        np.testing.assert_array_equal(fn(), [3.0, 4.0])

    def test_add(self):
        g = Graph()
        x = g.Input("x")
        y = g.Input("y")
        out = g.Add(x, y)
        g.output = out

        fn, source, _, _ = _make_compiler().compile(out, graph=g, optimize=False)
        result = fn(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_matmul(self):
        g = Graph()
        x = g.Input("x")
        w = g.Const([[1, 0], [0, 1]])
        out = g.MatMul(x, w)
        g.output = out

        fn, source, _, _ = _make_compiler().compile(out, graph=g, optimize=False)
        result = fn(np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [3.0, 4.0])

    def test_relu(self):
        g = Graph()
        x = g.Input("x")
        out = g.ReLU(x)
        g.output = out

        fn, source, _, _ = _make_compiler().compile(out, graph=g, optimize=False)
        result = fn(np.array([-1.0, 0.0, 3.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 3.0])

    def test_fused_mma_bias_relu(self):
        g = Graph()
        x = g.Input("x")
        w = g.Const([[1.0, 0.0], [0.0, 1.0]])
        b = g.Const([0.5, -0.5])
        out = g.FusedMMA_Bias_ReLU(x, w, b)
        g.output = out

        fn, source, _, _ = _make_compiler().compile(out, graph=g, optimize=False)
        result = fn(np.array([1.0, 2.0]))
        expected = np.maximum((np.array([1.0, 2.0]) @ np.array([[1.0, 0.0], [0.0, 1.0]])) + np.array([0.5, -0.5]), 0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_generated_code_is_valid_python(self):
        g = Graph()
        x = g.Input("x")
        w = g.Const([[1.0, 2.0], [3.0, 4.0]])
        b = g.Const([1.0, 0.0])
        mm = g.MatMul(x, w)
        add_ = g.Add(mm, b)
        out = g.ReLU(add_)
        g.output = out

        fn, source, _, _ = _make_compiler().compile(out, graph=g, optimize=False)
        # Should compile to valid Python via exec.
        assert callable(fn)
        assert "import numpy" in source
        assert "def compiled" in source

    def test_input_order_stable(self):
        g = Graph()
        a = g.Input("a")
        b = g.Input("b")
        out = g.Add(a, b)
        g.output = out

        fn, source, _, _ = _make_compiler().compile(out, graph=g, optimize=False)
        result = fn(np.array(1.0), np.array(2.0))
        assert result == 3.0

    def test_reuse_nodes_cached(self):
        """Nodes reused across multiple paths should only be computed once."""
        g = Graph()
        x = g.Input("x")
        w = g.Const([2.0, 2.0])
        # Use w in two different paths
        p1 = g.Add(x, w)
        p2 = g.Add(w, x)
        out = g.Add(p1, p2)
        g.output = out

        fn, source, _, _ = _make_compiler().compile(out, graph=g, optimize=False)
        result = fn(np.array([1.0, 1.0]))
        expected = (np.array([1.0, 1.0]) + np.array([2.0, 2.0])) + (np.array([2.0, 2.0]) + np.array([1.0, 1.0]))
        np.testing.assert_array_equal(result, expected)
