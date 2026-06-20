"""
NPU MLIR Backend — lowers TaskGraph through the MLIR lowering pipeline.

Replaces the pattern-match + ISAMapper path with:
  TaskGraph → linalg.matmul → tiling → npu.* → assembly → simulation
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import BackendError
from ..ir import TaskDesc, TaskGraph
from . import Backend


# Lazy imports: the mlir package is at the project-root level
_MLIR_IMPORTED = False


def _ensure_mlir():
    global _MLIR_IMPORTED
    if not _MLIR_IMPORTED:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _MLIR_IMPORTED = True


class NPUBackendMLIR(Backend):
    """Compiles TaskGraph through the MLIR lowering pipeline.

    Pipeline:
      TaskGraph → extract matmul shape → linalg.matmul
        → LinalgToTilesPass (4x4 decomposition)
        → TileToNpuOpsPass (npu.load/store/compute/barrier)
        → NpuOpsToProgramPass (assemble to 256x32-bit)
        → NPUSimulator (cycle-accurate execution)
        → result extraction

    Limitations:
      - Single matmul + optional relu only.
      - Input data must be representable as int32.
    """

    def __init__(self) -> None:
        self._config = None
        self._pipeline = None
        self._simulator = None

    @property
    def config(self):
        if self._config is None:
            _ensure_mlir()
            from mlir import HardwareConfig
            self._config = HardwareConfig()
        return self._config

    @property
    def pipeline(self):
        if self._pipeline is None:
            _ensure_mlir()
            from mlir import NPULoweringPipeline
            self._pipeline = NPULoweringPipeline(self.config)
        return self._pipeline

    @property
    def simulator(self):
        if self._simulator is None:
            _ensure_mlir()
            # Reuse mini-dl simulator for cycle-accurate execution
            try:
                from compiler.isa_mapper import NPUSimulator
                self._simulator = NPUSimulator()
            except ImportError:
                raise BackendError(
                    "mini-dl-compiler required for NPU simulator. "
                    "Install: pip install -e /path/to/mini-dl-compiler"
                ) from None
        return self._simulator

    def run(
        self,
        graph: TaskGraph,
        inputs: dict | None = None,
    ) -> Any:
        """Lower a TaskGraph through MLIR pipeline and execute on simulator.

        Args:
            graph: Compiled TaskGraph (SSA linear IR).
            inputs: Dict mapping Var name → nested list / scalar value.

        Returns:
            Computed result as nested list.

        Raises:
            BackendError: If the graph pattern is unsupported.
        """
        inputs = inputs or {}
        act_type = self._detect_pattern(graph)
        a_name, b_name = self._extract_matmul_inputs(graph)

        if a_name not in inputs or b_name not in inputs:
            raise BackendError(f"Missing inputs: {a_name}, {b_name}")

        A = np.asarray(inputs[a_name], dtype=np.int32)
        B = np.asarray(inputs[b_name], dtype=np.int32)

        if A.ndim != 2 or B.ndim != 2:
            raise BackendError("NPU matmul requires 2D inputs")

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise BackendError(f"Inner dimension mismatch: {K} vs {K2}")

        # Build linalg representation
        _ensure_mlir()
        from mlir import LinalgMatmul, ActType

        matmul = LinalgMatmul(
            lhs=a_name, rhs=b_name, result="%C",
            m=M, n=N, k=K,
        )

        act = ActType.RELU if act_type == "RELU" else ActType.NONE

        # Full MLIR lowering: linalg → tiles → npu → assembled program
        program = self.pipeline.lower(matmul, act=act)

        # Pack inputs into external memory
        ext_mem: dict[int, int] = {}
        offset_a, offset_b, offset_c = 0, 64, 128
        for i, v in enumerate(A.ravel()):
            ext_mem[offset_a + i] = int(v)
        for i, v in enumerate(B.ravel()):
            ext_mem[offset_b + i] = int(v)

        self.simulator.load_program(program)
        self.simulator.load_external_memory(ext_mem)
        self.simulator.run()

        results = self.simulator.read_external_memory(offset_c, M * N)
        C = np.array(results, dtype=np.int32).reshape(M, N)
        return C.tolist()

    # ── pattern detection ─────────────────────────────────────────

    @staticmethod
    def _detect_pattern(graph: TaskGraph) -> str:
        """Detect compute pattern: 'matmul' or 'matmul+relu'."""
        has_matmul = any(t.op == "matmul" for t in graph.tasks)
        has_relu = any(t.op == "relu" for t in graph.tasks)

        if not has_matmul:
            raise BackendError("NPU MLIR backend requires a matmul operation")

        return "RELU" if has_relu else "NONE"

    @staticmethod
    def _extract_matmul_inputs(graph: TaskGraph) -> tuple[str, str]:
        """Extract the two input variable names from the matmul task."""
        for task in graph.tasks:
            if task.op == "matmul":
                return task.inputs[0], task.inputs[1]
        raise BackendError("No matmul task found in graph")
