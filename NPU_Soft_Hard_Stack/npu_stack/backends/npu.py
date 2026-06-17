"""NPU backend — lowers TaskGraph to NPU ISA via the ISA Mapper.

Requires mini-dl-compiler to be installed (for ISAMapper).
"""

from __future__ import annotations

import sys
from typing import Any

from ..errors import BackendError
from ..ir import TaskDesc, TaskGraph
from . import Backend


class NPUBackend(Backend):
    """Compiles TaskGraph to NPU instructions and runs on the simulator.

    Limitations:
      - Only supports a single matmul + relu pattern.
      - Input data must be representable as int32.
    """

    def __init__(self) -> None:
        self._mapper = None
        self._simulator = None

    @property
    def mapper(self):
        """Lazy-load the ISA Mapper from mini-dl-compiler."""
        if self._mapper is None:
            try:
                from compiler.isa_mapper import ISAMapper
                self._mapper = ISAMapper()
            except ImportError:
                raise BackendError(
                    "mini-dl-compiler must be installed for NPU backend. "
                    "Run: pip install -e /path/to/mini-dl-compiler"
                ) from None
        return self._mapper

    @property
    def simulator(self):
        """Lazy-load the NPU simulator."""
        if self._simulator is None:
            try:
                from compiler.isa_mapper import NPUSimulator
                self._simulator = NPUSimulator()
            except ImportError:
                raise BackendError(
                    "mini-dl-compiler must be installed for NPU backend."
                ) from None
        return self._simulator

    def run(
        self,
        graph: TaskGraph,
        inputs: dict | None = None,
    ) -> Any:
        """Lower a TaskGraph to NPU ISA and execute on the simulator.

        Args:
            graph: Compiled TaskGraph.
            inputs: Dict mapping Var name → nested list / scalar value.

        Returns:
            Computed result as nested list.

        Raises:
            BackendError: If the graph cannot be lowered to NPU ISA.
        """
        inputs = inputs or {}

        # Pattern-match: matmul + relu
        matmul_task = None
        relu_task = None

        for task in graph.tasks:
            if task.op == "matmul":
                matmul_task = task
            elif task.op == "relu":
                relu_task = task

        if matmul_task is None:
            raise BackendError("NPU backend requires a matmul operation")

        # Get shapes from input data
        a_name = matmul_task.inputs[0]
        b_name = matmul_task.inputs[1]

        if a_name not in inputs or b_name not in inputs:
            raise BackendError(f"Missing inputs for matmul: {a_name}, {b_name}")

        import numpy as np
        A = np.asarray(inputs[a_name], dtype=np.int32)
        B = np.asarray(inputs[b_name], dtype=np.int32)

        if A.ndim != 2 or B.ndim != 2:
            raise BackendError("NPU matmul requires 2D inputs")

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise BackendError(f"Inner dim mismatch: {K} vs {K2}")

        act = "RELU" if relu_task is not None else "NONE"
        from compiler.isa_mapper.types import ActType
        act_type = ActType.RELU if relu_task else ActType.NONE

        # Map to NPU
        program = self.mapper.map_matmul(
            M=M, N=N, K=K,
            ext_a_addr=0,
            ext_b_addr=64,
            ext_c_addr=128,
            act=act_type,
        )

        # Pack inputs into external memory
        ext_mem: dict[int, int] = {}
        for i, v in enumerate(A.ravel()):
            ext_mem[i] = int(v)
        for i, v in enumerate(B.ravel()):
            ext_mem[64 + i] = int(v)

        self.simulator.load_program(program)
        self.simulator.load_external_memory(ext_mem)
        self.simulator.run()

        results = self.simulator.read_external_memory(128, M * N)
        C = np.array(results, dtype=np.int32).reshape(M, N)

        # Convert back to nested list for compatibility with CPU backend
        return C.tolist()
