#!/usr/bin/env python3
"""NPU test utilities: golden model generation and result verification.

Usage::

    from test_utils import golden_matmul, verify_npu_result
    golden = golden_matmul(A, B, act="relu")
    assert verify_npu_result(sim_output, golden)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def golden_matmul(
    A: NDArray[np.int32],
    B: NDArray[np.int32],
    act: str = "none",
) -> NDArray[np.int32]:
    """Compute the golden reference for C = A @ B (with optional activation).

    Args:
        A: Matrix A (M×K), int32.
        B: Matrix B (K×N), int32.
        act: Activation: ``"none"``, ``"relu"``.

    Returns:
        Matrix C (M×N), int32.
    """
    C = A.astype(np.int64) @ B.astype(np.int64)
    if act == "relu":
        C = np.maximum(C, 0)
    return C.astype(np.int32)


def verify_npu_result(
    actual: NDArray[np.int32],
    expected: NDArray[np.int32],
    rtol: float = 0.0,
    atol: int = 0,
) -> tuple[bool, str]:
    """Compare NPU output against golden reference.

    Returns:
        (passed, message)
    """
    if actual.shape != expected.shape:
        return False, f"Shape mismatch: {actual.shape} vs {expected.shape}"

    diff = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    max_diff = int(np.max(diff))
    if max_diff > atol:
        idx = np.unravel_index(int(np.argmax(diff)), actual.shape)
        return False, (
            f"Max difference {max_diff} at {idx}: "
            f"got {actual[idx]}, expected {expected[idx]}"
        )

    return True, "OK"


def generate_random_matmul(
    M: int = 4,
    N: int = 4,
    K: int = 4,
    value_range: tuple[int, int] = (0, 5),
    seed: int = 42,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], str]:
    """Generate random matrices and golden reference for testing.

    Returns:
        (A, B, C_golden, instruction_sequence_description)
    """
    rng = np.random.RandomState(seed)
    A = rng.randint(value_range[0], value_range[1], (M, K)).astype(np.int32)
    B = rng.randint(value_range[0], value_range[1], (K, N)).astype(np.int32)
    C = golden_matmul(A, B, act="relu")

    instr_desc = f"""\
MatMul {M}x{N}x{K} + ReLU
A ({M}x{K}): {A.shape} values @ ext_addr 0x0000
B ({K}x{N}): {B.shape} values @ ext_addr 0x0040
C ({M}x{N}): expected {C.shape} values @ ext_addr 0x0080"""

    return A, B, C, instr_desc


def generate_instruction_memory(
    operations: list[dict],
) -> list[int]:
    """Generate NPU instruction memory from a list of operation dicts.

    Each dict::

        {"op": "LOAD",  "bank": 0, "ext_addr": 0, "size": 16}
        {"op": "COMPUTE", "a_src": 0, "b_src": 1, "c_dst": 2,
         "mat_dim": "4x4", "act": "relu"}
        {"op": "STORE", "bank": 2, "ext_addr": 128, "size": 16}
        {"op": "BARRIER", "type": "dma"}
        {"op": "NOP"}

    Returns:
        List of 256 32-bit instructions.
    """
    OP_NOP, OP_LOAD, OP_STORE, OP_COMPUTE = 0, 1, 2, 3
    OP_BARRIER = 4
    BAR_DMA, BAR_COMPUTE = 1, 2

    instr_mem = [0] * 256
    idx = 0

    for op in operations:
        if op["op"] == "NOP":
            instr_mem[idx] = OP_NOP << 28
        elif op["op"] == "LOAD":
            instr_mem[idx] = (OP_LOAD << 28) | (op["bank"] << 26) | \
                             (op["size"] << 16) | (op["ext_addr"] & 0xFFFF)
        elif op["op"] == "STORE":
            instr_mem[idx] = (OP_STORE << 28) | (op["bank"] << 26) | \
                             (op["size"] << 16) | (op["ext_addr"] & 0xFFFF)
        elif op["op"] == "COMPUTE":
            mat_dim_map = {"4x4": 0, "2x2": 1, "1x1": 2}
            act_map = {"none": 0, "relu": 1}
            instr_mem[idx] = (OP_COMPUTE << 28) | \
                             (op["a_src"] << 26) | (op["b_src"] << 24) | \
                             (op["c_dst"] << 22) | \
                             (mat_dim_map.get(op.get("mat_dim", "4x4"), 0) << 20) | \
                             (act_map.get(op.get("act", "none"), 0) << 18)
        elif op["op"] == "BARRIER":
            bar_type = BAR_DMA if op.get("type") == "dma" else BAR_COMPUTE
            instr_mem[idx] = (OP_BARRIER << 28) | (bar_type << 26)
        idx += 1

    return instr_mem


if __name__ == "__main__":
    # Quick self-test
    A, B, C, desc = generate_random_matmul(4, 4, 4)
    print(desc)
    print("C_golden:")
    print(C)
    print()

    instrs = generate_instruction_memory([
        {"op": "LOAD", "bank": 0, "ext_addr": 0, "size": 16},
        {"op": "LOAD", "bank": 1, "ext_addr": 64, "size": 16},
        {"op": "BARRIER", "type": "dma"},
        {"op": "COMPUTE", "a_src": 0, "b_src": 1, "c_dst": 2,
         "mat_dim": "4x4", "act": "relu"},
        {"op": "BARRIER", "type": "compute"},
        {"op": "STORE", "bank": 2, "ext_addr": 128, "size": 16},
        {"op": "BARRIER", "type": "dma"},
        {"op": "NOP"},
    ])
    print(f"Instructions: {len(instrs)} words")
    for i, w in enumerate(instrs[:10]):
        print(f"  [{i:3d}] 0x{w:08X}")
