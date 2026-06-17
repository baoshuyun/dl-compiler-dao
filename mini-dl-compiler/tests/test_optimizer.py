"""Unit tests for the optimization passes."""

import numpy as np
import pytest

from compiler.ir import Graph, Node
from compiler.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph():
    return Graph()


def _const(g, val, name=""):
    return g.Const(val, name=name)


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

class TestConstantFolding:
    def test_fold_const_add(self):
        g = _make_graph()
        out = g.Add(_const(g, [2, 2, 2]), _const(g, [3, 3, 3]))
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result.op == "const"
        assert np.array_equal(result.value, [5, 5, 5])
        assert "[fold]" in opt.logs[0]

    def test_fold_const_matmul(self):
        g = _make_graph()
        a = _const(g, [[1, 2], [3, 4]])
        b = _const(g, [[1, 0], [0, 1]])
        out = g.MatMul(a, b)
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result.op == "const"
        assert np.array_equal(result.value, [[1, 2], [3, 4]])

    def test_fold_add_zero_left(self):
        g = _make_graph()
        x = g.Input("x")
        out = g.Add(_const(g, 0), x)
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result is x

    def test_fold_add_zero_right(self):
        g = _make_graph()
        x = g.Input("x")
        out = g.Add(x, _const(g, 0))
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result is x

    def test_fold_relu_const(self):
        g = _make_graph()
        out = g.ReLU(_const(g, [-1, 0, 3]))
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result.op == "const"
        assert np.array_equal(result.value, [0, 0, 3])

    def test_fold_relu_const_positive(self):
        g = _make_graph()
        out = g.ReLU(_const(g, [5, 10]))
        opt = Optimizer(g)
        result = opt.fold(out)
        assert np.array_equal(result.value, [5, 10])

    def test_fold_input_unchanged(self):
        g = _make_graph()
        x = g.Input("x")
        opt = Optimizer(g)
        result = opt.fold(x)
        assert result is x

    def test_fold_const_unchanged(self):
        g = _make_graph()
        c = _const(g, [1, 2, 3])
        opt = Optimizer(g)
        result = opt.fold(c)
        assert result is c

    def test_fold_deep_nested(self):
        g = _make_graph()
        # ((c1 + c2) + c3)  ->  const
        s1 = g.Add(_const(g, [1]), _const(g, [2]))
        s2 = g.Add(s1, _const(g, [3]))
        opt = Optimizer(g)
        result = opt.fold(s2)
        assert result.op == "const"
        assert np.array_equal(result.value, [6])

    def test_fold_non_const_add_preserved(self):
        g = _make_graph()
        x = g.Input("x")
        c = _const(g, [1, 2, 3])
        out = g.Add(x, c)
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result.op == "add"
        assert result.inputs[0] is x
        assert result.inputs[1] is c

    def test_fold_partial_add_zero(self):
        """x + 0 should simplify to x even when x is not const."""
        g = _make_graph()
        x = g.Input("x")
        out = g.Add(x, _const(g, 0))
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result is x

    def test_fold_fused_mma_all_const(self):
        g = _make_graph()
        x = _const(g, [[1, 0]])
        w = _const(g, [[2], [3]])
        b = _const(g, [-1])
        out = g.FusedMMA_Bias_ReLU(x, w, b)
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result.op == "const"
        # x(1x2) @ w(2x1) + b(1,) = 1x1 result
        assert np.array_equal(result.value, [[1]])

    def test_fold_fused_mma_partial_const(self):
        g = _make_graph()
        x = g.Input("x")
        w = _const(g, [[1, 2], [3, 4]])
        b = _const(g, [0, 0])
        out = g.FusedMMA_Bias_ReLU(x, w, b)
        opt = Optimizer(g)
        result = opt.fold(out)
        assert result.op == "fused_mma_bias_relu"


# ---------------------------------------------------------------------------
# Operator fusion
# ---------------------------------------------------------------------------

class TestFusion:
    def test_fuse_matmul_add_relu(self):
        g = _make_graph()
        x = g.Input("x")
        w = g.Const([[1, 2], [3, 4]])
        b = g.Const([0, 0])
        mm = g.MatMul(x, w)
        add_ = g.Add(mm, b)
        out = g.ReLU(add_)
        opt = Optimizer(g)
        result = opt.fuse(out)
        assert result.op == "fused_mma_bias_relu"
        assert result.inputs[0] is x
        assert result.inputs[1] is w
        assert result.inputs[2] is b

    def test_fuse_add_matmul_relu(self):
        """bias + matmul order should also fuse."""
        g = _make_graph()
        x = g.Input("x")
        w = g.Const([[1, 2], [3, 4]])
        b = g.Const([0, 0])
        mm = g.MatMul(x, w)
        add_ = g.Add(b, mm)  # reversed order
        out = g.ReLU(add_)
        opt = Optimizer(g)
        result = opt.fuse(out)
        assert result.op == "fused_mma_bias_relu"

    def test_fuse_leaf_unchanged(self):
        g = _make_graph()
        x = g.Input("x")
        opt = Optimizer(g)
        result = opt.fuse(x)
        assert result is x

    def test_fuse_no_pattern_no_change(self):
        """relu(add(x, y)) without matmul should not fuse."""
        g = _make_graph()
        x = g.Input("x")
        y = g.Input("y")
        add_ = g.Add(x, y)
        out = g.ReLU(add_)
        opt = Optimizer(g)
        result = opt.fuse(out)
        assert result.op == "relu"
        assert result.inputs[0].op == "add"


# ---------------------------------------------------------------------------
# Dead code elimination
# ---------------------------------------------------------------------------

class TestDCE:
    def test_dce_removes_dead_nodes(self):
        g = _make_graph()
        x = g.Input("x")
        y = g.Input("y")
        out = g.Add(x, g.Const(1))
        # y is never used — it should be removed
        initial_count = len(g.nodes)
        opt = Optimizer(g)
        opt.dce(out)
        assert y not in g.nodes
        assert len(g.nodes) < initial_count

    def test_dce_keeps_all_reachable(self):
        g = _make_graph()
        x = g.Input("x")
        w = g.Const([1, 2, 3])
        out = g.Add(x, w)
        opt = Optimizer(g)
        opt.dce(out)
        assert x in g.nodes
        assert w in g.nodes
        assert out in g.nodes

    def test_dce_no_dead_nodes(self):
        g = _make_graph()
        x = g.Input("x")
        g.output = x
        opt = Optimizer(g)
        opt.dce(x)
        assert "[dce] no dead" in opt.logs[0]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_run_full_pipeline(self):
        g = _make_graph()
        x = g.Input("x")
        w = g.Const([[1, 0], [0, 1]])
        b = g.Const([1, 2])  # non-zero so add is not eliminated by fold
        mm = g.MatMul(x, w)
        add_ = g.Add(mm, b)
        out = g.ReLU(add_)
        _ = g.Const([99])  # dead
        g.output = out

        opt = Optimizer(g)
        result = opt.run()
        assert result.op == "fused_mma_bias_relu"
        # Dead const should be gone.
        for n in g.nodes:
            assert n.op != "const" or not np.array_equal(n.value, [99])
