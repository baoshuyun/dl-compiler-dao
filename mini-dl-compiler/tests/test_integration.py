"""End-to-end integration tests for the compiler pipeline."""

import numpy as np

from compiler.codegen import NumpyCompiler
from compiler.ir import Graph


class TestEndToEnd:
    def test_demo_model_correctness(self):
        """The demo model from main.py should produce correct output."""
        g = Graph()
        x = g.Input("x")
        W = g.Const([[1, 2, 3], [4, 5, 6]], name="W")
        b = g.Const([1, 0, -1], name="b")
        c = g.Add(
            g.Const([2, 2, 2], name="c1"),
            g.Const([3, 3, 3], name="c2"),
        )
        dead = g.ReLU(g.MatMul(g.Const([1, 2]), g.Const([[9], [9]])), name="dead")
        mm = g.MatMul(x, W)
        y = g.ReLU(g.Add(mm, g.Add(b, c)))
        g.output = y

        compiler = NumpyCompiler()
        fn, src, opt_node, logs = compiler.compile(y, graph=g)

        x_val = np.array([1.0, 2.0])
        result = fn(x_val)

        # Eager reference
        W_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b_np = np.array([1, 0, -1], dtype=np.float64)
        c_np = np.array([5, 5, 5], dtype=np.float64)
        expected = np.maximum((x_val @ W_np) + (b_np + c_np), 0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_layer(self):
        """Two-layer feed-forward network."""
        g = Graph()
        x = g.Input("x")
        w1 = g.Const([[0.5, -0.5], [1.0, 0.0]])
        b1 = g.Const([0.1, 0.2])
        h = g.ReLU(g.Add(g.MatMul(x, w1), b1))
        w2 = g.Const([[1.0], [-1.0]])
        b2 = g.Const([0.0])
        out = g.Add(g.MatMul(h, w2), b2)
        g.output = out

        compiler = NumpyCompiler()
        fn, src, opt_node, logs = compiler.compile(out, graph=g)

        x_val = np.array([2.0, -1.0])
        result = fn(x_val)

        # Eager
        h_eager = np.maximum(x_val @ np.array([[0.5, -0.5], [1.0, 0.0]]) + np.array([0.1, 0.2]), 0)
        expected = h_eager @ np.array([[1.0], [-1.0]]) + np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_no_dead_code_in_output(self):
        """After optimization + DCE, generated code should not contain dead consts."""
        g = Graph()
        x = g.Input("x")
        _ = g.Const([999.0])  # dead
        out = g.ReLU(x)
        g.output = out

        compiler = NumpyCompiler()
        _, src, _, _ = compiler.compile(out, graph=g)
        assert "999" not in src

    def test_const_folding_reduces_node_count(self):
        g = Graph()
        x = g.Input("x")
        # All-const sub-graph: (1+2) + 3 = 6
        s1 = g.Add(g.Const([1]), g.Const([2]))
        s2 = g.Add(s1, g.Const([3]))
        out = g.Add(x, s2)
        g.output = out

        compiler = NumpyCompiler()
        _, src, opt_node, logs = compiler.compile(out, graph=g)

        # The folded const [6] should appear in source
        assert "6" in src
        # Optimization should have folded the chain
        fold_logs = [l for l in logs if "[fold]" in l]
        assert len(fold_logs) >= 2

    def test_output_type_is_numpy(self):
        g = Graph()
        x = g.Input("x")
        out = g.ReLU(x)
        g.output = out

        compiler = NumpyCompiler()
        fn, _, _, _ = compiler.compile(out, graph=g, optimize=False)
        result = fn(np.array([-1.0, 2.0]))
        assert isinstance(result, np.ndarray)
