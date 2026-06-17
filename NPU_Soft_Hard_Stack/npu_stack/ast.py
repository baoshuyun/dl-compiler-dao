"""AST (Abstract Syntax Tree) for the NPU compiler stack.

Defines the expression nodes that users compose to build neural-network
computation graphs.  Nodes support operator overloading for ergonomic
graph construction.

Example::

    from npu_stack.ast import Var, relu
    expr = relu(Var("x") * Var("w") + Var("b"))
"""

from __future__ import annotations

from typing import Any


# ════════════════════════════════════════════════════════════════
# Base
# ════════════════════════════════════════════════════════════════

class Expr:
    """Base expression node with operator overloading."""

    def __add__(self, other: Expr | Any) -> Add:
        return Add(self, ensure_expr(other))

    def __radd__(self, other: Expr | Any) -> Add:
        return Add(ensure_expr(other), self)

    def __mul__(self, other: Expr | Any) -> Mul:
        return Mul(self, ensure_expr(other))

    def __rmul__(self, other: Expr | Any) -> Mul:
        return Mul(ensure_expr(other), self)


# ════════════════════════════════════════════════════════════════
# Leaf nodes
# ════════════════════════════════════════════════════════════════

class Var(Expr):
    """Named input variable."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Var name must be non-empty")
        self.name = name

    def __repr__(self) -> str:
        return f"Var({self.name!r})"

    def __hash__(self) -> int:
        return hash(("var", self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and self.name == other.name


class Const(Expr):
    """Literal constant value."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Const({self.value!r})"

    def __hash__(self) -> int:
        return hash(("const", id(self)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Const) and self.value == other.value


# ════════════════════════════════════════════════════════════════
# Binary ops
# ════════════════════════════════════════════════════════════════

class Add(Expr):
    """Element-wise addition: left + right."""

    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left!r} + {self.right!r})"

    def __hash__(self) -> int:
        return hash(("add", self.left, self.right))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Add) and
                self.left == other.left and self.right == other.right)


class Mul(Expr):
    """Element-wise multiplication: left * right."""

    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left!r} * {self.right!r})"

    def __hash__(self) -> int:
        return hash(("mul", self.left, self.right))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Mul) and
                self.left == other.left and self.right == other.right)


class MatMul(Expr):
    """Matrix multiplication: left @ right.

    Shapes: left(M,K) @ right(K,N) → result(M,N).
    """

    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"MatMul({self.left!r}, {self.right!r})"

    def __hash__(self) -> int:
        return hash(("matmul", self.left, self.right))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, MatMul) and
                self.left == other.left and self.right == other.right)


# ════════════════════════════════════════════════════════════════
# Unary ops
# ════════════════════════════════════════════════════════════════

class Relu(Expr):
    """ReLU activation: max(0, x)."""

    def __init__(self, x: Expr) -> None:
        self.x = x

    def __repr__(self) -> str:
        return f"Relu({self.x!r})"

    def __hash__(self) -> int:
        return hash(("relu", self.x))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Relu) and self.x == other.x


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def ensure_expr(x: Any) -> Expr:
    """Promote a raw Python value to a Const node."""
    if isinstance(x, Expr):
        return x
    return Const(x)


def relu(x: Any) -> Relu:
    """Build a ReLU node, auto-promoting the argument."""
    return Relu(ensure_expr(x))


def matmul(left: Any, right: Any) -> MatMul:
    """Build a MatMul node."""
    return MatMul(ensure_expr(left), ensure_expr(right))
