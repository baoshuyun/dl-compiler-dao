"""Shape inference for AST expressions.

Infers the output shape of each AST node at compile time so that
shape mismatches are caught before execution.
"""

from __future__ import annotations

from . import ast
from .errors import ShapeError


def infer_shape(expr: ast.Expr) -> tuple[int, ...]:
    """Infer the output shape of an AST expression.

    Returns a tuple of dimension sizes.  Scalars are represented as
    the empty tuple ``()``.

    Raises:
        ShapeError: If shapes are incompatible.
    """
    if isinstance(expr, ast.Var) or isinstance(expr, ast.Const):
        # Shapes are not known at AST level for Var/Const;
        # callers should pass concrete shapes via infer_shape_with_inputs.
        return ()

    if isinstance(expr, ast.Add):
        left = infer_shape(expr.left)
        right = infer_shape(expr.right)
        if left and right and left != right:
            raise ShapeError(
                f"Add shape mismatch: {left} vs {right}")
        return left or right

    if isinstance(expr, ast.Mul):
        left = infer_shape(expr.left)
        right = infer_shape(expr.right)
        if left and right and left != right:
            raise ShapeError(
                f"Mul shape mismatch: {left} vs {right}")
        return left or right

    if isinstance(expr, ast.MatMul):
        left = infer_shape(expr.left)
        right = infer_shape(expr.right)
        if len(left) != 2 or len(right) != 2:
            raise ShapeError(
                f"MatMul requires 2D inputs, got {left} and {right}")
        if left[1] != right[0]:
            raise ShapeError(
                f"MatMul inner dim mismatch: {left[1]} vs {right[0]}")
        return (left[0], right[1])

    if isinstance(expr, ast.Relu):
        return infer_shape(expr.x)

    raise ShapeError(f"Unknown AST node: {type(expr).__name__}")


def infer_shape_with_inputs(
    expr: ast.Expr,
    input_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    """Infer shape with known input shapes.

    Args:
        expr: AST root.
        input_shapes: Dict mapping Var name → shape tuple.

    Returns:
        Inferred output shape.
    """
    if isinstance(expr, ast.Var):
        if expr.name not in input_shapes:
            raise ShapeError(f"Unknown input variable: {expr.name!r}")
        return input_shapes[expr.name]

    if isinstance(expr, ast.Const):
        import numpy as np
        val = np.asarray(expr.value)
        return tuple(val.shape) if val.ndim > 0 else ()

    if isinstance(expr, ast.Add):
        left = infer_shape_with_inputs(expr.left, input_shapes)
        right = infer_shape_with_inputs(expr.right, input_shapes)
        if left and right and left != right:
            raise ShapeError(f"Add shape mismatch: {left} vs {right}")
        return left or right

    if isinstance(expr, ast.Mul):
        left = infer_shape_with_inputs(expr.left, input_shapes)
        right = infer_shape_with_inputs(expr.right, input_shapes)
        if left and right and left != right:
            raise ShapeError(f"Mul shape mismatch: {left} vs {right}")
        return left or right

    if isinstance(expr, ast.MatMul):
        left = infer_shape_with_inputs(expr.left, input_shapes)
        right = infer_shape_with_inputs(expr.right, input_shapes)
        if len(left) != 2 or len(right) != 2:
            raise ShapeError(
                f"MatMul requires 2D inputs, got {left} and {right}")
        if left[1] != right[0]:
            raise ShapeError(
                f"MatMul inner dim mismatch: {left[1]} vs {right[0]}")
        return (left[0], right[1])

    if isinstance(expr, ast.Relu):
        return infer_shape_with_inputs(expr.x, input_shapes)

    raise ShapeError(f"Unknown AST node: {type(expr).__name__}")
