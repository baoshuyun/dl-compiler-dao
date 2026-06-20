"""
Frontend lowering: Soft_Stack TaskGraph → linalg dialect.

Bridges the gap between the expression-level AST/IR and the
MLIR lowering pipeline. Takes a compiled TaskGraph and extracts
linalg operations for downstream lowering to NPU instructions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .npu_dialect import ActType
from .lowering_pipeline import LinalgMatmul


@dataclass
class LinalgProgram:
    """A program expressed in the linalg dialect.

    Contains one or more linalg operations ready for lowering.
    For now: single matmul with optional activation.
    """
    matmul: LinalgMatmul
    act: ActType = ActType.NONE


class TaskGraphToLinalg:
    """Lower Soft_Stack TaskGraph to linalg dialect.

    Pattern-matches the TaskGraph for matmul + optional relu
    and constructs a LinalgMatmul with shape information.
    """

    def lower(self, graph, inputs: dict) -> LinalgProgram:
        """Convert a TaskGraph to a LinalgProgram.

        Args:
            graph: Soft_Stack TaskGraph with tasks list.
            inputs: Dict mapping variable names to numpy arrays or lists.

        Returns:
            LinalgProgram ready for NPU lowering pipeline.

        Raises:
            ValueError: If the graph pattern is not supported.
        """
        import numpy as np

        # Find the compute op (mul or matmul) and optional relu
        compute_task = None
        relu_task = None
        for task in graph.tasks:
            if task.op in ("matmul", "mul"):
                compute_task = task
            elif task.op == "relu":
                relu_task = task

        if compute_task is None:
            raise ValueError("TaskGraph must contain a mul or matmul operation")

        a_name = compute_task.inputs[0]
        b_name = compute_task.inputs[1]
        out_name = compute_task.output

        # Infer shapes from input data
        A = np.asarray(inputs[a_name], dtype=np.int32)
        B = np.asarray(inputs[b_name], dtype=np.int32)

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("NPU matmul requires 2D inputs")

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"Inner dimension mismatch: {K} vs {K2}")

        act = ActType.RELU if relu_task else ActType.NONE

        linalg_op = LinalgMatmul(
            lhs=a_name,
            rhs=b_name,
            result=out_name,
            m=M, n=N, k=K,
        )

        return LinalgProgram(matmul=linalg_op, act=act)
