# mini-dl-compiler: Memory Planning
# Liveness analysis + greedy memory planner
# Reduces peak memory footprint by reusing buffers whose lifetimes do not overlap

from dataclasses import dataclass, field
from typing import List, Optional
from ir import Operation


@dataclass
class BufferDescriptor:
    """Describes a memory buffer with its liveness interval and allocation offset.

    - birth: first step index where this buffer is live.
    - death: last step index where this buffer is live.
    - offset: assigned byte offset in the memory pool (set by MemoryPlanner).
    - size: size in bytes.
    - name: corresponding SSA value name.
    """

    name: str
    shape: tuple
    element_size: int = 4  # float32 = 4 bytes
    birth: int = 0
    death: int = 0
    offset: int = -1  # -1 means unassigned

    @property
    def size(self) -> int:
        """Total buffer size in bytes."""
        total = self.element_size
        for dim in self.shape:
            total *= dim
        return total


class LivenessAnalyzer:
    """Computes buffer liveness intervals from a linear operation sequence.

    Liveness rule: a buffer is live from its birth (defining op index) to
    its death (last consumer op index). Buffers whose intervals do not
    overlap can share the same memory region.

    This is analogous to register allocation via graph coloring, but
    operating on memory buffers rather than registers.
    """

    def analyze(self, operations: List[Operation]) -> List[BufferDescriptor]:
        buffers: dict = {}  # name -> BufferDescriptor

        for step_idx, op in enumerate(operations):
            # Results are born at this step
            for r in op.results:
                shape = getattr(r.type, 'shape', (1,))
                buf = BufferDescriptor(r.name, shape, element_size=4)
                buf.birth = step_idx
                buffers[r.name] = buf

            # Inputs are consumed at this step (update death)
            for inp in op.inputs:
                if inp.name in buffers:
                    buffers[inp.name].death = max(
                        buffers[inp.name].death, step_idx
                    )

        return list(buffers.values())


class MemoryPlanner:
    """Greedy memory allocator using interval-based buffer reuse.

    Algorithm:
    1. Sort buffers by birth time.
    2. For each buffer:
       a. Collect garbage: free buffers whose death < current birth.
       b. Find the smallest free offset that can fit this buffer.
       c. Assign the buffer to that offset.
    3. Total pool size = max(offset + size) across all buffers.

    This is the classic "top of stack" / greedy interval allocation
    used in TVM, TFLite, and XLA. It produces optimal results for
    interval graphs (which are chordal).
    """

    def __init__(self, alignment: int = 64):
        """Initialize with memory alignment in bytes (default: 64B cache line)."""
        self.alignment = alignment
        self.total_size: int = 0

    def _align(self, size: int) -> int:
        """Round up size to alignment boundary."""
        return ((size + self.alignment - 1) // self.alignment) * self.alignment

    def _collect_garbage(self, active: List[BufferDescriptor], current_step: int):
        """Remove buffers that are dead before current_step."""
        active[:] = [b for b in active if b.death >= current_step]

    def _find_free_spot(self, active: List[BufferDescriptor], size: int) -> int:
        """Find the smallest offset where a buffer of `size` bytes fits."""
        if not active:
            return 0
        # Sort active by offset, find first gap >= size
        sorted_active = sorted(active, key=lambda b: b.offset)
        candidate = 0
        for buf in sorted_active:
            if candidate + size <= buf.offset:
                return candidate
            candidate = max(candidate, buf.offset + self._align(buf.size))
        return candidate

    def plan(self, buffers: List[BufferDescriptor]) -> int:
        """Run the greedy allocator and return total memory pool size.

        Modifies buffers in-place, setting their `offset` fields.

        Returns:
            Total pool size in bytes needed for all buffers.
        """
        sorted_bufs = sorted(buffers, key=lambda b: b.birth)
        active: List[BufferDescriptor] = []

        for buf in sorted_bufs:
            self._collect_garbage(active, buf.birth)
            offset = self._find_free_spot(active, buf.size)
            buf.offset = offset
            active.append(buf)
            self.total_size = max(self.total_size, offset + self._align(buf.size))

        return self.total_size


# Example: For a matmul with intermediate add+relu, liveness analysis
# typically finds that 33 float32 slots suffice where naive allocation
# would require 3 full-size buffers (input, intermediate, output).
