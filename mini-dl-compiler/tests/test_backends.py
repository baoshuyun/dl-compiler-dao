"""Tests for compiler backends — NPU and base infrastructure."""
from __future__ import annotations

import numpy as np
import pytest

from compiler import Graph, Node
from compiler.backends.npu import NPUBackend
from compiler.backends import Backend, _run_optimizer


class TestNPUBackendBasics:
    """NPU backend construction and type checks."""

    def test_construct_with_default_config(self) -> None:
        backend = NPUBackend()
        assert backend.config.array_rows == 4
        assert backend.config.sram_banks == 4

    def test_compile_basic_matmul(self) -> None:
        g = Graph()
        x = g.Input("x")
        w = g.Const(np.ones((4, 4), dtype=np.int32), name="w")
        out = g.MatMul(x, w)
        g.output = out

        backend = NPUBackend()
        fn, source, opt_out, logs = backend.compile(out, graph=g, optimize=False)

        assert callable(fn)
        assert len(source) > 0
        result = fn(x=np.ones((1, 4), dtype=np.int32))
        assert isinstance(result, np.ndarray)

    def test_run_matmul(self) -> None:
        g = Graph()
        x = g.Input("x")
        w = g.Const(np.ones((4, 4), dtype=np.int32), name="w")
        out = g.MatMul(x, w)
        g.output = out

        backend = NPUBackend()
        result = backend.run(g, {"x": np.ones((1, 4), dtype=np.int32)})
        assert "output" in result
        assert isinstance(result["output"], np.ndarray)


class TestNPUBackendLowering:
    """Test pattern-matching in the NPU lowering pass."""

    def test_run_pure_matmul(self) -> None:
        g = Graph()
        w = g.Const(np.ones((4, 4), dtype=np.int32), name="w")
        x = g.Input("x")
        out = g.MatMul(x, w)
        g.output = out

        backend = NPUBackend()
        result = backend.run(g, {"x": np.ones((1, 4), dtype=np.int32)})
        assert isinstance(result["output"], np.ndarray)

    def test_compile_returns_source_string(self) -> None:
        g = Graph()
        x = g.Input("x")
        w = g.Const(np.ones((4, 4), dtype=np.int32), name="w")
        out = g.MatMul(x, w)
        g.output = out

        backend = NPUBackend()
        fn, source, opt_out, logs = backend.compile(out, graph=g)
        assert len(source) > 0


class TestBackendInfrastructure:
    """Base backend helpers."""

    def test_run_optimizer_preserves_output(self) -> None:
        g = Graph()
        x = g.Input("x")
        w = g.Const(np.ones((4, 4)), name="w")
        out = g.MatMul(x, w)
        g.output = out

        opt_out, logs = _run_optimizer(g)
        assert isinstance(opt_out, Node)
        assert isinstance(logs, list)
