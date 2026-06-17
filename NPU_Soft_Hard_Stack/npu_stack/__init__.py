"""NPU_Soft_Hard_Stack — a minimal DL compiler frontend.

Pipeline::

    AST expression  →  Compiler  →  TaskGraph IR  →  Backend (CPU / NPU)

Example::

    from npu_stack import Var, relu, compile_and_run

    expr = relu(Var("x") * Var("w") + Var("b"))
    result = compile_and_run(expr, {"x": [1, 2], "w": [3, 4], "b": 0})
"""

from __future__ import annotations

from .ast import Var, Const, Add, Mul, MatMul, Relu, relu, matmul, ensure_expr
from .ir import Compiler, TaskDesc, TaskGraph
from .backends import Backend, CPUBackend
from .shape import infer_shape, infer_shape_with_inputs
from .errors import CompilerError, ShapeError, BackendError


def compile_and_run(expr, inputs: dict | None = None, *, backend: str = "cpu"):
    """End-to-end convenience: compile and run on CPU or NPU.

    Args:
        expr: AST expression.
        inputs: Dict mapping Var name → value.
        backend: Backend name: ``"cpu"`` or ``"npu"``.

    Returns:
        Computed result.
    """
    graph = Compiler().compile(expr)

    if backend == "cpu":
        return CPUBackend().run(graph, inputs)
    elif backend == "npu":
        from .backends.npu import NPUBackend
        return NPUBackend().run(graph, inputs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'cpu' or 'npu'.")


__all__ = [
    "Var", "Const", "Add", "Mul", "MatMul", "Relu",
    "relu", "matmul", "ensure_expr",
    "Compiler", "TaskDesc", "TaskGraph",
    "Backend", "CPUBackend",
    "infer_shape", "infer_shape_with_inputs",
    "CompilerError", "ShapeError", "BackendError",
    "compile_and_run",
]
