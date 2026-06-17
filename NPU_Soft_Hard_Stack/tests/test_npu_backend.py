"""Test NPU backend: Soft_Stack → ISAMapper → NPU Simulator → result.

Proves the three projects are fully interoperable at the ISA level.
"""

from __future__ import annotations

import pytest

from npu_stack import Var, relu, matmul, compile_and_run, TaskGraph, Compiler
from npu_stack.backends.npu import NPUBackend
from npu_stack.errors import BackendError


class TestNPUBackend:
    """Verify Soft_Stack NPU backend produces correct results via ISAMapper."""

    def test_matmul_relu_via_isamapper(self) -> None:
        """End-to-end: AST → Compiler → ISAMapper → Simulator → correct result."""
        # Build: C = ReLU(X @ W)
        expr = relu(matmul(Var("x"), Var("w")))

        # Inputs: 4x4 identity
        x_data = [[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]]
        w_data = [[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]]

        graph = Compiler().compile(expr)
        backend = NPUBackend()
        result = backend.run(graph, {"x": x_data, "w": w_data})

        # Golden: C = ReLU(I @ I) = all-4s
        assert result == [[4, 4, 4, 4],
                          [4, 4, 4, 4],
                          [4, 4, 4, 4],
                          [4, 4, 4, 4]], f"Got {result}"

    def test_npu_and_cpu_match(self) -> None:
        """NPU and CPU backends must produce the same result."""
        x = [[2, 3], [4, 1]]
        w = [[1, 0], [0, 1]]

        expr = relu(matmul(Var("x"), Var("w")))
        graph = Compiler().compile(expr)

        cpu_result = compile_and_run(expr, {"x": x, "w": w}, backend="cpu")
        npu_result = NPUBackend().run(graph, {"x": x, "w": w})

        assert cpu_result == npu_result, \
            f"CPU={cpu_result} vs NPU={npu_result}"

    def test_rejects_non_matmul(self) -> None:
        """NPU backend should reject graphs without matmul."""
        expr = Var("a") + Var("b")
        graph = Compiler().compile(expr)

        with pytest.raises(BackendError, match="matmul"):
            NPUBackend().run(graph, {"a": 1, "b": 2})
