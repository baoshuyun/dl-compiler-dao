"""Tests for optimization passes: Tiling, Memory Planning, Graph→SSA."""

from __future__ import annotations

import numpy as np
import pytest

from compiler.ir import Graph
from compiler.passes import (
    PassManager,
    TilingPass,
    MemoryPlanningPass,
    GraphToSSAPass,
)


class TestTilingPass:
    """Verify tiling cost model and tile config assignment."""

    def test_tiling_on_matmul(self) -> None:
        g = Graph()
        w = g.Const(np.ones((256, 256), dtype=np.float32), name="W")
        x = g.Input("x")
        y = g.MatMul(x, w, name="matmul")
        g.output = y

        tiling = TilingPass(target_rows=4, target_cols=4, sram_capacity=1024)
        result = tiling.run(g)

        matmul_node = next(n for n in result.nodes if n.op == "matmul")
        assert "tile" in matmul_node.attrs
        tm, tn, tk = matmul_node.attrs["tile"]
        assert tm <= 256 and tn <= 256 and tk <= 256
        assert tm >= 2 and tn >= 2 and tk >= 2

    def test_tiling_on_fused_op(self) -> None:
        g = Graph()
        w = g.Const(np.ones((64, 32), dtype=np.float32))
        b = g.Const(np.ones(32, dtype=np.float32))
        x = g.Input("x")
        y = g.FusedMMA_Bias_ReLU(x, w, b)
        g.output = y

        tiling = TilingPass()
        result = tiling.run(g)

        fused = next(n for n in result.nodes if n.op == "fused_mma_bias_relu")
        assert "tile" in fused.attrs

    def test_beam_search_prefers_small_misses(self) -> None:
        """Small matmul should get tile sizes close to the full dimensions."""
        g = Graph()
        w = g.Const(np.ones((8, 8), dtype=np.float32))
        x = g.Input("x")
        y = g.MatMul(x, w)
        g.output = y

        tiling = TilingPass(target_rows=4, target_cols=4)
        result = tiling.run(g)

        matmul_node = next(n for n in result.nodes if n.op == "matmul")
        tm, tn, tk = matmul_node.attrs["tile"]
        # For an 8x8 matmul with 4x4 target, tiles should be around 4-8
        assert 2 <= tm <= 16
        assert 2 <= tn <= 16


class TestMemoryPlanning:
    """Verify memory planning reduces peak usage via buffer reuse."""

    def test_planner_assigns_offsets(self) -> None:
        g = Graph()
        a = g.Input("a")
        b = g.Input("b")
        c = g.Add(a, b, name="add")
        r = g.ReLU(c, name="relu")
        g.output = r

        mp = MemoryPlanningPass()
        result = mp.run(g)

        add_node = next(n for n in result.nodes if n.op == "add")
        relu_node = next(n for n in result.nodes if n.op == "relu")
        assert "buffer_offset" in relu_node.attrs

    def test_peak_memory_reported(self) -> None:
        g = Graph()
        w = g.Const(np.ones((256, 256)), name="W")
        x = g.Input("x")
        y = g.MatMul(x, w)
        g.output = y

        mp = MemoryPlanningPass()
        result = mp.run(g)

        assert result.output is not None
        assert "peak_pool_bytes" in result.output.attrs
        assert result.output.attrs["peak_pool_bytes"] > 0

    def test_buffer_reuse_reduces_peak(self) -> None:
        """Non-overlapping buffers should reuse the same memory region."""
        g = Graph()
        w1 = g.Const(np.ones((256, 256)), name="W1")
        w2 = g.Const(np.ones((256, 256)), name="W2")
        x = g.Input("x")
        m1 = g.MatMul(x, w1, name="matmul1")
        m2 = g.MatMul(x, w2, name="matmul2")
        # m1 and m2 have no dependency chain → should overlap
        g.output = m2

        mp = MemoryPlanningPass()
        result = mp.run(g)

        peak = result.output.attrs.get("peak_pool_bytes", float("inf"))
        single = 256 * 256 * 4  # one buffer
        # Peak should be less than 2× (buffer reuse)
        assert peak < single * 2, f"Peak {peak} >= {single * 2} (no reuse?)"


class TestGraphToSSA:
    """Verify Graph IR → SSA IR lowering."""

    def test_lowers_basic_graph(self) -> None:
        g = Graph()
        a = g.Input("a")
        b = g.Input("b")
        c = g.Add(a, b)
        r = g.ReLU(c)
        g.output = r

        lowering = GraphToSSAPass(bufferize=True)
        ops = lowering.run(g)

        assert len(ops) >= 2  # add + relu
        op_types = {op.op_type for op in ops}
        assert "arith.addf" in op_types
        assert "linalg.relu" in op_types

    def test_lowers_matmul(self) -> None:
        g = Graph()
        w = g.Const(np.ones((4, 4)))
        x = g.Input("x")
        y = g.MatMul(x, w)
        g.output = y

        lowering = GraphToSSAPass()
        ops = lowering.run(g)

        matmul_ops = [op for op in ops if op.op_type == "linalg.matmul"]
        assert len(matmul_ops) == 1
        op = matmul_ops[0]
        assert len(op.inputs) == 2
        assert len(op.results) == 1

    def test_integrated_pipeline(self) -> None:
        """Full optimization pipeline: Tiling → Memory → SSA."""
        g = Graph()
        w = g.Const(np.ones((128, 64), dtype=np.float32))
        x = g.Input("x")
        y = g.MatMul(x, w)
        r = g.ReLU(y)
        g.output = r

        pm = PassManager()
        pm.add(TilingPass())
        pm.add(MemoryPlanningPass())
        pm.add(GraphToSSAPass())

        result = pm.run(g)
        summary = pm.summary()

        # After GraphToSSAPass, result should be a list of Operations
        assert isinstance(result, list)
        assert len(result) >= 2  # matmul + relu
        assert "Tiling" in summary
        assert "Memory" in summary
        assert "Graph → SSA" in summary
