"""CPU reference backend using an in-memory Python interpreter."""

from __future__ import annotations

from typing import Any

from ..ir import TaskDesc, TaskGraph
from . import Backend


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _map_unary(x: Any, fn) -> Any:
    if _is_seq(x):
        return type(x)(_map_unary(v, fn) for v in x)
    return fn(x)


def _map_binary(a: Any, b: Any, fn) -> Any:
    """Element-wise binary op with scalar broadcasting."""
    if _is_seq(a) and _is_seq(b):
        if len(a) != len(b):
            raise ValueError(f"Shape mismatch: {len(a)} != {len(b)}")
        return type(a)(_map_binary(x, y, fn) for x, y in zip(a, b))
    if _is_seq(a):
        return type(a)(_map_binary(x, b, fn) for x in a)
    if _is_seq(b):
        return type(b)(_map_binary(a, y, fn) for y in b)
    return fn(a, b)


class Executor:
    """In-memory interpreter for TaskGraph."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def reset(self) -> None:
        self._values.clear()

    def set_value(self, name: str, value: Any) -> None:
        self._values[name] = value

    def get_value(self, name: str) -> Any:
        if name not in self._values:
            raise KeyError(f"Variable {name!r} not found")
        return self._values[name]

    def run(self, tasks: list[TaskDesc], inputs: dict | None = None) -> None:
        if inputs:
            for k, v in inputs.items():
                self.set_value(k, v)
        for task in tasks:
            self.execute(task)

    def execute(self, task: TaskDesc) -> Any:
        op = task.op.lower()

        if op == "const":
            self.set_value(task.output, task.attrs["value"])

        elif op == "add":
            a = self.get_value(task.inputs[0])
            b = self.get_value(task.inputs[1])
            self.set_value(task.output, _map_binary(a, b, lambda x, y: x + y))

        elif op == "mul":
            a = self.get_value(task.inputs[0])
            b = self.get_value(task.inputs[1])
            self.set_value(task.output, _map_binary(a, b, lambda x, y: x * y))

        elif op == "matmul":
            a = self.get_value(task.inputs[0])
            b = self.get_value(task.inputs[1])
            self.set_value(task.output, _dot(a, b))

        elif op == "sub":
            a = self.get_value(task.inputs[0])
            b = self.get_value(task.inputs[1])
            self.set_value(task.output, _map_binary(a, b, lambda x, y: x - y))

        elif op == "relu":
            x = self.get_value(task.inputs[0])
            self.set_value(task.output, _map_unary(x, lambda v: max(v, 0)))

        else:
            raise ValueError(f"Unknown op: {op!r}")

        return self.get_value(task.output)


def _dot(a: Any, b: Any) -> Any:
    """Simple nested-list matrix multiplication."""
    if _is_seq(a) and _is_seq(b):
        if _is_seq(a[0]) and _is_seq(b[0]):
            # 2D @ 2D
            rows_a, cols_a = len(a), len(a[0])
            rows_b, cols_b = len(b), len(b[0])
            if cols_a != rows_b:
                raise ValueError(
                    f"MatMul inner dim mismatch: {cols_a} vs {rows_b}")
            result = [[0.0] * cols_b for _ in range(rows_a)]
            for i in range(rows_a):
                for j in range(cols_b):
                    s = 0.0
                    for k in range(cols_a):
                        s += a[i][k] * b[k][j]
                    result[i][j] = s
            return result
    raise ValueError(f"MatMul requires 2D inputs")


class CPUBackend(Backend):
    """Reference CPU backend."""

    def __init__(self) -> None:
        self.executor = Executor()

    def run(self, graph: TaskGraph, inputs: dict | None = None) -> Any:
        self.executor.reset()
        self.executor.run(graph.tasks, inputs=inputs)
        return self.executor.get_value(graph.output)
