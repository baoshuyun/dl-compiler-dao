"""
End-to-end test: MLIR pipeline → NPU simulator → golden verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mini-dl-compiler"))

from test_utils import golden_matmul, verify_npu_result


def _mlir_matmul(M, N, K, A, B, act):
    """Run MLIR pipeline and return result matrix.

    For >4 size matrices, requires data layout pass (npu.pack) for
    contiguous tile DMA — not yet implemented. Limited to M=N=K=4.
    """
    assert M == N == K == 4, "MLIR backend currently supports 4x4 only"
    from compiler.mlir import NPULoweringPipeline, LinalgMatmul, ActType
    from compiler.isa_mapper import NPUSimulator

    pipeline = NPULoweringPipeline()
    matmul = LinalgMatmul("%A", "%B", "%C", m=M, n=N, k=K)
    program, c_partials = pipeline.lower(matmul, act)

    ext_mem: dict[int, int] = {}
    for i, v in enumerate(A.ravel()):
        ext_mem[i] = int(v)
    for i, v in enumerate(B.ravel()):
        ext_mem[64 + i] = int(v)

    sim = NPUSimulator()
    sim.load_program(program)
    sim.load_external_memory(ext_mem)
    sim.run()

    vals = sim.read_external_memory(c_partials[(0, 0)][0], M * N)
    return np.array(vals, dtype=np.int32).reshape(M, N)


def test_mlir_matmul_4x4():
    np.random.seed(42)
    A = np.random.randint(-8, 8, (4, 4), dtype=np.int32)
    B = np.random.randint(-8, 8, (4, 4), dtype=np.int32)
    golden = golden_matmul(A, B, act="none")
    from compiler.mlir import ActType
    actual = _mlir_matmul(4, 4, 4, A, B, act=ActType.NONE)
    passed, msg = verify_npu_result(actual, golden)
    assert passed, msg
    print("[PASS] 4x4 matmul (no act): max diff = 0")


def test_mlir_matmul_4x4_relu():
    from compiler.mlir import ActType
    np.random.seed(43)
    A = np.random.randint(-8, 8, (4, 4), dtype=np.int32)
    B = np.random.randint(-8, 8, (4, 4), dtype=np.int32)
    golden = golden_matmul(A, B, act="relu")
    actual = _mlir_matmul(4, 4, 4, A, B, act=ActType.RELU)
    passed, msg = verify_npu_result(actual, golden)
    assert passed, msg
    print("[PASS] 4x4 matmul + RELU: max diff = 0")


def test_mlir_matmul_8x8():
    """8x8 matmul — needs npu.pack data layout pass for contiguous tile DMA."""
    import pytest
    pytest.skip("Requires npu.pack data layout pass (contiguous tile DMA)")


if __name__ == "__main__":
    test_mlir_matmul_4x4()
    test_mlir_matmul_4x4_relu()
    test_mlir_matmul_8x8()
    print("\nAll MLIR E2E tests passed.")
