"""Compiler: AST → TaskGraph linear SSA IR.

Walks an AST expression tree and emits a linear sequence of
TaskDesc instructions in SSA form (temporary variable naming).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import ast
from .errors import CompilerError


# ════════════════════════════════════════════════════════════════
# IR types
# ════════════════════════════════════════════════════════════════

@dataclass
class TaskDesc:
    """A single instruction in the linear IR.

    Attributes:
        op: Operation type (const, add, mul, matmul, relu, sub).
        inputs: SSA variable names consumed by this task.
        output: SSA variable name produced by this task.
        attrs: Opaque metadata dictionary.
    """
    op: str
    inputs: list[str] = field(default_factory=list)
    output: str = ""
    attrs: dict = field(default_factory=dict)


@dataclass
class TaskGraph:
    """Linear SSA IR.

    Attributes:
        tasks: Ordered list of TaskDesc instructions.
        output: Name of the SSA variable holding the final result.
    """
    tasks: list[TaskDesc]
    output: str

    def __repr__(self) -> str:
        lines = [f"TaskGraph(output={self.output!r}, {len(self.tasks)} tasks)"]
        for t in self.tasks:
            in_str = ", ".join(t.inputs)
            lines.append(f"  {t.output} = {t.op}({in_str})")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Compiler
# ════════════════════════════════════════════════════════════════

class Compiler:
    """Walks an AST and emits a linear TaskGraph in SSA form.

    Usage::

        compiler = Compiler()
        graph = compiler.compile(expr)
        print(graph)
    """

    def __init__(self) -> None:
        self._temp_idx = 0

    def _new_temp(self) -> str:
        self._temp_idx += 1
        return f"%t{self._temp_idx}"

    def compile(self, expr: ast.Expr) -> TaskGraph:
        """Lower an AST expression to a TaskGraph."""
        self._temp_idx = 0
        tasks: list[TaskDesc] = []
        out = self._compile_expr(expr, tasks)
        return TaskGraph(tasks=tasks, output=out)

    def _compile_expr(self, expr: ast.Expr, tasks: list[TaskDesc]) -> str:
        if isinstance(expr, ast.Var):
            return expr.name

        if isinstance(expr, ast.Const):
            out = self._new_temp()
            tasks.append(TaskDesc(
                op="const", output=out,
                attrs={"value": expr.value}))
            return out

        if isinstance(expr, ast.Add):
            left = self._compile_expr(expr.left, tasks)
            right = self._compile_expr(expr.right, tasks)
            out = self._new_temp()
            tasks.append(TaskDesc(
                op="add", inputs=[left, right], output=out))
            return out

        if isinstance(expr, ast.Mul):
            left = self._compile_expr(expr.left, tasks)
            right = self._compile_expr(expr.right, tasks)
            out = self._new_temp()
            tasks.append(TaskDesc(
                op="mul", inputs=[left, right], output=out))
            return out

        if isinstance(expr, ast.MatMul):
            left = self._compile_expr(expr.left, tasks)
            right = self._compile_expr(expr.right, tasks)
            out = self._new_temp()
            tasks.append(TaskDesc(
                op="matmul", inputs=[left, right], output=out))
            return out

        if isinstance(expr, ast.Relu):
            x = self._compile_expr(expr.x, tasks)
            out = self._new_temp()
            tasks.append(TaskDesc(
                op="relu", inputs=[x], output=out))
            return out

        raise CompilerError(f"Unknown expression type: {type(expr).__name__}")
