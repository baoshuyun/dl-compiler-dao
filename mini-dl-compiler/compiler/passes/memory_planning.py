"""Memory planning pass — liveness analysis + buffer reuse.

Analyses the lifetime of each intermediate tensor in the computation
graph and assigns them to shared memory buffers, reducing peak memory
usage via inter-buffer reuse.

Algorithm: interval-based greedy allocation (first-fit with liveness).
This is the same approach used in TFLite, TVM, and XLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ir import Graph, Node
from . import Pass


@dataclass
class BufferDesc:
    """Describes a logical buffer in the computation graph."""
    name: str
    size_bytes: int
    birth: int   # step index when buffer is first written
    death: int   # step index after which buffer is no longer needed
    node: Node | None = None
    offset: int = 0  # assigned by planner


class MemoryPlanningPass(Pass):
    """Liveness-aware memory planner.

    Reuses memory across non-overlapping buffer lifetimes to reduce
    peak allocation.

    Attributes:
        alignment: Byte alignment for allocations (default 64 for cache line).
        element_bytes: Bytes per element for default type (default 4 for FP32).
    """

    name = "Memory Planning"

    def __init__(
        self,
        alignment: int = 64,
        element_bytes: int = 4,
    ) -> None:
        self.alignment = alignment
        self.element_bytes = element_bytes

    def run(self, ir: Graph, **kwargs: Any) -> Graph:
        """Analyse liveness and assign buffer offsets.

        Each node that produces a tensor gets ``buffer_offset`` and
        ``buffer_size`` attributes.  The total pool size is stored
        as ``total_pool_bytes`` in the graph's first node attrs
        (or via a synthetic marker).
        """
        if ir.output is None:
            return ir

        # 1. Topological order (inputs first)
        try:
            order = ir.topological_order(ir.output)
        except RecursionError:
            return ir

        # 2. Liveness: assign birth/death to each node's output
        step: dict[str, int] = {}
        for i, node in enumerate(order):
            step[id(node)] = i

        buffers: list[BufferDesc] = []
        for i, node in enumerate(order):
            if node.op in ("input", "const"):
                continue  # input/const buffers managed externally

            # Compute output size
            size = self._estimate_size(node)
            if size == 0:
                continue

            # Death = max step of all consumers
            death = i
            for other in order:
                if node in other.inputs:
                    idx = step.get(id(other), i)
                    death = max(death, idx)

            buffers.append(BufferDesc(
                name=node.name or f"{node.op}_{i}",
                size_bytes=size * self.element_bytes,
                birth=i,
                death=death,
                node=node,
            ))

        if not buffers:
            return ir

        # 3. Greedy allocation (first-fit by birth time)
        buffers.sort(key=lambda b: b.birth)
        active: list[BufferDesc] = []  # currently live buffers
        free_list: list[tuple[int, int]] = [(0, 2 ** 30)]  # (base, limit)

        for buf in buffers:
            # GC dead buffers
            active = [b for b in active if b.death >= buf.birth]
            # Rebuild free list from active intervals
            used = sorted((b.offset, b.offset + b.size_bytes) for b in active)
            free_list = []
            cursor = 0
            for base, limit in used:
                if cursor < base:
                    free_list.append((cursor, base))
                cursor = max(cursor, limit)
            free_list.append((cursor, 2 ** 30))

            # Allocate from first-fit free region
            for base, limit in free_list:
                if limit - base >= buf.size_bytes:
                    aligned = (base + self.alignment - 1) // self.alignment * self.alignment
                    if aligned + buf.size_bytes <= limit:
                        buf.offset = aligned
                        break

            active.append(buf)

            # Annotate the node
            if buf.node:
                buf.node.attrs["buffer_offset"] = buf.offset
                buf.node.attrs["buffer_size"] = buf.size_bytes

        # 4. Compute peak
        if active:
            peak = max(b.offset + b.size_bytes for b in active)
            # Store in output node attrs
            if ir.output:
                ir.output.attrs["peak_pool_bytes"] = peak
                ir.output.attrs["num_buffers"] = len(buffers)

        return ir

    def verify(self, before: Graph, after: Graph) -> list[str]:
        warnings = []
        if after.output and "peak_pool_bytes" in after.output.attrs:
            peak = after.output.attrs["peak_pool_bytes"]
            if peak > 16384:  # 16 KB warning
                warnings.append(
                    f"Peak memory {peak} bytes exceeds typical SRAM (16 KB)"
                )
        return warnings

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_size(node: Node) -> int:
        """Estimate the number of elements in a node's output."""
        if isinstance(node.value, np.ndarray):
            return int(node.value.size)
        if "shape" in node.attrs:
            s = node.attrs["shape"]
            n = 1
            for d in s:
                n *= d
            return n
        # Heuristic: default to 256 elements
        return 256
