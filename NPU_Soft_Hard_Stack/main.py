#!/usr/bin/env python3
"""NPU_Soft_Hard_Stack — End-to-end demo.

Pipeline: AST → Compiler → TaskGraph → Backend (CPU / NPU)
"""

from __future__ import annotations

from npu_stack import Var, relu, matmul, compile_and_run


def demo_cpu() -> None:
    """CPU backend demos."""
    print("=" * 50)
    print("  NPU_Soft_Hard_Stack — CPU Backend Demo")
    print("=" * 50)

    # Example 1: Scalar arithmetic
    print("\n-- Example 1: (a + b) * c --")
    a, b, c = Var("a"), Var("b"), Var("c")
    result = compile_and_run((a + b) * c, {"a": 3, "b": 4, "c": 2})
    print(f"  (3 + 4) * 2 = {result}")
    assert result == 14

    # Example 2: ReLU
    print("\n-- Example 2: relu(a + b) --")
    result = compile_and_run(relu(a + b), {"a": -5, "b": 3})
    print(f"  relu(-5 + 3) = {result}")
    assert result == 0

    # Example 3: Vector broadcast
    print("\n-- Example 3: (x + y) * z — broadcast --")
    x, y, z = Var("x"), Var("y"), Var("z")
    result = compile_and_run((x + y) * z,
                             {"x": [1, 2, 3], "y": [4, 5, 6], "z": 2})
    print(f"  ([1,2,3] + [4,5,6]) * 2 = {result}")
    assert result == [10, 14, 18]

    # Example 4: MLP affine + ReLU
    print("\n-- Example 4: relu(w * x + b) --")
    w, x2, b2 = Var("w"), Var("x"), Var("b")
    result = compile_and_run(relu(w * x2 + b2),
                             {"w": [2, 0, -1], "x": [1, 2, 3], "b": [1, 1, 1]})
    print(f"  relu([2,0,-1]*[1,2,3] + [1,1,1]) = {result}")
    assert result == [3, 1, 0]

    print("\n" + "=" * 50)
    print("  All CPU examples passed!")
    print("=" * 50)


def demo_npu() -> None:
    """NPU backend demo — uses ISAMapper + NPU simulator."""
    print("\n" + "=" * 50)
    print("  NPU_Soft_Hard_Stack — NPU Backend Demo")
    print("=" * 50)

    x = Var("x")
    w = Var("w")

    # Example 5: MatMul + ReLU on NPU
    print("\n-- Example 5: relu(x @ w) on NPU --")
    expr = relu(matmul(x, w))

    x_data = [[1, 1, 1, 1],
              [2, 2, 2, 2],
              [1, 1, 1, 1],
              [2, 2, 2, 2]]
    w_data = [[1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1]]

    result = compile_and_run(expr, {"x": x_data, "w": w_data}, backend="npu")
    print(f"  relu(X @ I):\n{result}")

    # row0: sum(1,1,1,1)=4,   matmul with all-1s → 4 per element
    # row1: sum(2,2,2,2)=8,   → 8 per element
    expected = [[4, 4, 4, 4],
                [8, 8, 8, 8],
                [4, 4, 4, 4],
                [8, 8, 8, 8]]
    assert result == expected, f"NPU mismatch: got {result}, expected {expected}"

    print("\n  NPU backend verified against golden reference!")
    print("=" * 50)


if __name__ == "__main__":
    demo_cpu()
    demo_npu()
