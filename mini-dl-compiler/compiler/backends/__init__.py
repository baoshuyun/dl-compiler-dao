"""Compiler backends.

Each backend translates the optimized IR graph into executable code
for a specific target platform.

Available backends:
    numpy  — NumPy reference (portable CPU)
    npu    — NPU 32-bit ISA via the ISA Mapper
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from ..ir import Graph, Node
from ..optimizer import Optimizer


class Backend(ABC):
    """Abstract base for compiler backends."""

    @abstractmethod
    def compile(
        self,
        output: Node,
        *,
        graph: Graph | None = None,
        optimize: bool = True,
    ) -> tuple[Callable[..., Any], str, Node, list[str]]:
        """Compile a graph into a callable function.

        Returns:
            (callable, source_code, optimized_output, logs)
        """
        ...

    @abstractmethod
    def run(
        self,
        graph: Graph,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Optimize, compile, and execute a graph with the given inputs.

        Returns:
            Dict mapping output names to numpy arrays.
        """
        ...


def _run_optimizer(graph: Graph) -> tuple[Node, list[str]]:
    """Run the standard optimization pipeline."""
    opt = Optimizer(graph)
    output = opt.run()
    return output, opt.logs
