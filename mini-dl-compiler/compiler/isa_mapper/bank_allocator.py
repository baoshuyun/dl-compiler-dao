"""SRAM bank allocator.

Assigns logical buffers to one of the 4 physical SRAM banks using
a liveness-aware greedy strategy.  Supports ping/pong double-buffering
for overlapping DMA transfers with compute.

Bank convention (matches NPU_Project `npu_top.v`):
  bank0 — Matrix A ping
  bank1 — Matrix B ping
  bank2 — Matrix A pong / result scratchpad
  bank3 — Matrix B pong / result scratchpad
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import BufferAssignment, HardwareConfig


@dataclass
class _LiveRange:
    """Internal: tracks when a buffer is born and dies in the schedule."""
    name: str
    num_words: int
    first_use: int   # step index where buffer is first read/written
    last_use: int    # step index after which buffer is no longer needed
    is_input: bool = False
    is_output: bool = False


class BankAllocator:
    """Greedy first-fit allocator with liveness analysis.

    Usage::

        alloc = BankAllocator(config)
        assignments = alloc.allocate(live_ranges)
    """

    def __init__(self, config: HardwareConfig | None = None) -> None:
        self.config = config or HardwareConfig()
        self.bank_depth = self.config.bank_depth

    # ── Public API ───────────────────────────────────────────────

    def allocate(
        self,
        live_ranges: list[_LiveRange],
    ) -> dict[str, BufferAssignment]:
        """Assign each live range to a bank and offset.

        Strategy:
          1. Input buffers A, B → banks 0, 1 (ping).
          2. Output buffers → banks 2, 3 (or reuse input banks after last use).
          3. Intermediate buffers → first-fit across all 4 banks.
          4. Use liveness to reuse space: a dead buffer's space can be
             given to a new buffer.

        Returns:
            Dict mapping buffer name → BufferAssignment.
        """
        # Per-bank free list: list of (base, limit) free regions
        banks: list[list[tuple[int, int]]] = [
            [(0, self.bank_depth)] for _ in range(self.config.sram_banks)
        ]

        # Per-bank active allocations: [(name, base, limit, last_use)]
        active: list[list[tuple[str, int, int, int]]] = [[] for _ in range(self.config.sram_banks)]

        assignments: dict[str, BufferAssignment] = {}

        for buf in sorted(live_ranges, key=lambda b: b.first_use):
            # 1. Garbage-collect dead buffers before allocating
            for bank_idx in range(self.config.sram_banks):
                self._gc(bank_idx, buf.first_use, banks, active)

            # 2. Pick bank
            preferred = self._preferred_banks(buf)
            bank_idx = self._find_bank(buf.num_words, banks, preferred)

            if bank_idx is None:
                # Fallback: try any bank
                bank_idx = self._find_bank(buf.num_words, banks, None)

            if bank_idx is None:
                raise RuntimeError(
                    f"Cannot allocate {buf.name} ({buf.num_words} words): "
                    f"all banks full at step {buf.first_use}"
                )

            # 3. Allocate from the chosen bank
            base = self._alloc_from_bank(buf.num_words, banks[bank_idx])
            assignments[buf.name] = BufferAssignment(
                name=buf.name,
                bank=bank_idx,
                base_addr=base,
                num_words=buf.num_words,
            )
            active[bank_idx].append((buf.name, base, base + buf.num_words, buf.last_use))

        return assignments

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _preferred_banks(buf: _LiveRange) -> list[int] | None:
        """Return preferred bank indices for a buffer, or None for any."""
        name_lower = buf.name.lower()
        if "a" == name_lower or name_lower.startswith("a_") or name_lower.startswith("input_a"):
            return [0, 2]  # ping/pong for A
        if "b" == name_lower or name_lower.startswith("b_") or name_lower.startswith("input_b"):
            return [1, 3]  # ping/pong for B
        if buf.is_output:
            return [2, 3]  # output banks
        return None  # any bank

    @staticmethod
    def _find_bank(
        num_words: int,
        banks: list[list[tuple[int, int]]],
        preferred: list[int] | None,
    ) -> int | None:
        """Find the first bank (preferring *preferred*) with a free region ≥ num_words."""
        candidates = preferred if preferred is not None else list(range(len(banks)))
        for bank_idx in candidates:
            for base, limit in banks[bank_idx]:
                if limit - base >= num_words:
                    return bank_idx
        return None

    @staticmethod
    def _alloc_from_bank(num_words: int, free_list: list[tuple[int, int]]) -> int:
        """Allocate *num_words* from the first-fitting free region. Returns base address."""
        for i, (base, limit) in enumerate(free_list):
            if limit - base >= num_words:
                free_list[i] = (base + num_words, limit)
                if free_list[i][0] >= free_list[i][1]:
                    free_list.pop(i)
                return base
        raise RuntimeError("No free region found (inconsistent state)")

    @staticmethod
    def _gc(
        bank_idx: int,
        current_step: int,
        banks: list[list[tuple[int, int]]],
        active: list[list[tuple[str, int, int, int]]],
    ) -> None:
        """Reclaim space from buffers whose last_use < current_step."""
        new_active = []
        freed_regions = []
        for name, base, limit, last_use in active[bank_idx]:
            if last_use < current_step:
                freed_regions.append((base, limit))
            else:
                new_active.append((name, base, limit, last_use))
        active[bank_idx] = new_active
        # Merge freed regions back into free list
        for region in freed_regions:
            banks[bank_idx].append(region)
        banks[bank_idx].sort(key=lambda r: r[0])
        # Coalesce adjacent regions
        merged: list[tuple[int, int]] = []
        for base, limit in banks[bank_idx]:
            if merged and merged[-1][1] == base:
                merged[-1] = (merged[-1][0], limit)
            else:
                merged.append((base, limit))
        banks[bank_idx][:] = merged
