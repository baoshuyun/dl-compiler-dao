"""Tests for NPU_Soft_Hard_Stack."""

from __future__ import annotations

import sys
import pytest

from npu_stack import (
    Var, Const, Add, Mul, MatMul, Relu,
    relu, matmul, ensure_expr,
    Compiler, TaskGraph, CPUBackend,
    infer_shape_with_inputs, compile_and_run,
    ShapeError,
)


class TestAST:
    def test_var_creation(self) -> None:
        v = Var("x")
        assert v.name == "x"

    def test_operator_overloading(self) -> None:
        expr = Var("a") + Var("b")
        assert isinstance(expr, Add)
        assert isinstance(expr.left, Var)
        assert isinstance(expr.right, Var)

    def test_relu_helper(self) -> None:
        expr = relu(Var("x"))
        assert isinstance(expr, Relu)

    def test_matmul_helper(self) -> None:
        expr = matmul(Var("a"), Var("b"))
        assert isinstance(expr, MatMul)

    def test_const_promotion(self) -> None:
        expr = Var("x") + 5
        assert isinstance(expr.right, Const)
        assert expr.right.value == 5

    def test_complex_expression(self) -> None:
        expr = relu(Var("x") * Var("w") + Var("b"))
        assert isinstance(expr, Relu)
        assert isinstance(expr.x, Add)


class TestCompiler:
    def test_compiles_var(self) -> None:
        compiler = Compiler()
        graph = compiler.compile(Var("x"))
        assert graph.output == "x"
        assert len(graph.tasks) == 0

    def test_compiles_const(self) -> None:
        compiler = Compiler()
        graph = compiler.compile(Const(42))
        assert len(graph.tasks) == 1
        assert graph.tasks[0].op == "const"

    def test_compiles_add(self) -> None:
        compiler = Compiler()
        graph = compiler.compile(Var("a") + Var("b"))
        assert len(graph.tasks) == 1
        assert graph.tasks[0].op == "add"
        assert graph.tasks[0].inputs == ["a", "b"]

    def test_compiles_matmul(self) -> None:
        compiler = Compiler()
        graph = compiler.compile(matmul(Var("x"), Var("w")))
        assert any(t.op == "matmul" for t in graph.tasks)

    def test_compiles_mlp(self) -> None:
        compiler = Compiler()
        graph = compiler.compile(relu(Var("x") * Var("w") + Var("b")))
        ops = {t.op for t in graph.tasks}
        assert "mul" in ops
        assert "add" in ops
        assert "relu" in ops


class TestCPUExecution:
    def test_scalar_computation(self) -> None:
        result = compile_and_run(
            Var("a") + Var("b"),
            {"a": 3, "b": 4},
        )
        assert result == 7

    def test_vector_broadcast(self) -> None:
        result = compile_and_run(
            (Var("x") + Var("y")) * Var("z"),
            {"x": [1, 2, 3], "y": [4, 5, 6], "z": 2},
        )
        assert result == [10, 14, 18]

    def test_relu_negative(self) -> None:
        result = compile_and_run(
            relu(Var("a") + Var("b")),
            {"a": -5, "b": 3},
        )
        assert result == 0

    def test_relu_positive(self) -> None:
        result = compile_and_run(
            relu(Var("a") + Var("b")),
            {"a": 5, "b": 3},
        )
        assert result == 8

    def test_mlp_pattern(self) -> None:
        result = compile_and_run(
            relu(Var("w") * Var("x") + Var("b")),
            {"w": [3, 0], "x": [1, 2], "b": 0},
        )
        # Mul(w, x) = [3, 0], +b = [3, 0], relu = [3, 0]
        assert result == [3, 0]


class TestShapeInference:
    def test_infers_scalar(self) -> None:
        shape = infer_shape_with_inputs(
            Var("a") + Var("b"),
            {"a": (1,), "b": (1,)},
        )
        assert shape == (1,)

    def test_infers_matmul_shape(self) -> None:
        shape = infer_shape_with_inputs(
            matmul(Var("x"), Var("w")),
            {"x": (4, 8), "w": (8, 4)},
        )
        assert shape == (4, 4)

    def test_detects_matmul_mismatch(self) -> None:
        with pytest.raises(ShapeError, match="inner dim"):
            infer_shape_with_inputs(
                matmul(Var("x"), Var("w")),
                {"x": (4, 8), "w": (4, 4)},
            )

    def test_detects_add_mismatch(self) -> None:
        with pytest.raises(ShapeError, match="shape mismatch"):
            infer_shape_with_inputs(
                Var("a") + Var("b"),
                {"a": (4,), "b": (8,)},
            )
