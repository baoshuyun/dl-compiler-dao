"""Linear scheduler with automatic BARRIER insertion.

Given a sequence of tiled operations, this module resolves data
dependencies and inserts BARRIER instructions to guarantee correct
execution order on the NPU hardware.

Key invariants enforced:
  1. A COMPUTE that reads a buffer must wait for the LOAD that fills it.
  2. A COMPUTE that reads a buffer must wait for the previous COMPUTE
     that wrote it (RAW hazard).
  3. A STORE that reads a buffer must wait for the COMPUTE that produced it.
  4. A LOAD that overwrites a buffer must wait for the last reader of
     the old contents (WAW hazard via DMA barrier).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import BarrierType, ScheduledOp, TileOp


class Scheduler:
    """Resolves data dependencies and inserts BARRIER markers.

    Usage::

        sched = Scheduler()
        scheduled = sched.schedule(tiled_ops, dma_ops)
    """

    def __init__(self) -> None:
        self._last_writer: dict[str, str] = {}   # buffer_name -> "compute"|"dma"
        self._last_reader: dict[str, int] = {}    # buffer_name -> step_index
        self._dma_pending: bool = False
        self._compute_pending: bool = False

    def schedule(
        self,
        tiled_ops: list[TileOp],
        extra_ops: list[dict[str, Any]] | None = None,
        *,
        interleaved: list[dict[str, Any] | TileOp] | None = None,
    ) -> list[ScheduledOp]:
        """Produce a linear schedule with BARRIERs inserted.

        Args:
            tiled_ops: Ordered list of tiled compute operations
                       (ignored if *interleaved* is provided).
            extra_ops: DMA/metadata ops
                       (ignored if *interleaved* is provided).
            interleaved: Pre-interleaved list of dict (DMA) and TileOp.
                         Preserves exact ordering from caller.

        Returns:
            Ordered list of ScheduledOp with barriers populated.
        """
        result: list[ScheduledOp] = []

        # Build index: for each buffer, track (step_idx, is_writer, kind)
        buf_accesses: dict[str, list[tuple[int, bool, str]]] = defaultdict(list)
        all_ops: list[tuple[int, Any, str]] = []

        if interleaved is not None:
            # Use caller-provided interleaved order
            for op in interleaved:
                idx = len(all_ops)
                if isinstance(op, dict):
                    kind = op["type"]
                    all_ops.append((idx, op, kind))
                    buf_name = op["buf"]
                    is_writer = kind == "load"
                    buf_accesses[buf_name].append((idx, is_writer, kind))
                else:
                    tile = op
                    all_ops.append((idx, tile, "compute"))
                    buf_accesses[tile.c_name].append((idx, True, "compute"))
                    buf_accesses[tile.a_name].append((idx, False, "compute"))
                    buf_accesses[tile.b_name].append((idx, False, "compute"))
        else:
            # Legacy mode: all DMA ops first, then all compute ops
            dma_ops = extra_ops or []
            for dma in dma_ops:
                idx = len(all_ops)
                kind = dma["type"]
                all_ops.append((idx, dma, kind))
                buf_name = dma["buf"]
                is_writer = kind == "load"
                buf_accesses[buf_name].append((idx, is_writer, kind))
            for tile in tiled_ops:
                idx = len(all_ops)
                all_ops.append((idx, tile, "compute"))
                buf_accesses[tile.c_name].append((idx, True, "compute"))
                buf_accesses[tile.a_name].append((idx, False, "compute"))
                buf_accesses[tile.b_name].append((idx, False, "compute"))

        # Walk through ops in order, inserting barriers where needed
        for idx, op, kind in all_ops:
            if kind == "compute":
                assert isinstance(op, TileOp)
                barriers: list[BarrierType] = []

                # Check if inputs need DMA barrier
                for input_buf in (op.a_name, op.b_name):
                    if self._needs_barrier(input_buf, idx, buf_accesses):
                        if BarrierType.DMA not in barriers:
                            barriers.append(BarrierType.DMA)
                            self._dma_pending = False

                # Check if we need compute barrier (previous compute writes our input)
                for input_buf in (op.a_name, op.b_name):
                    w = self._last_writer.get(input_buf)
                    if w == "compute" and self._compute_pending:
                        if BarrierType.COMPUTE not in barriers:
                            barriers.append(BarrierType.COMPUTE)
                            self._compute_pending = False
                        break

                result.append(ScheduledOp(
                    op=op,
                    barriers_before=barriers,
                ))
                self._last_writer[op.c_name] = "compute"
                self._compute_pending = True

            elif kind in ("load", "store"):
                barriers: list[BarrierType] = []
                buf_name = op["buf"]

                if kind == "store":
                    # Must wait for compute to finish if reading compute output
                    w = self._last_writer.get(buf_name)
                    if w == "compute" and self._compute_pending:
                        barriers.append(BarrierType.COMPUTE)
                        self._compute_pending = False

                if kind == "load":
                    # Must wait for DMA if previous DMA not done
                    if self._dma_pending:
                        barriers.append(BarrierType.DMA)
                        self._dma_pending = False
                    # Must wait for compute if we're overwriting a buffer being read
                    w = self._last_writer.get(buf_name)
                    if w == "compute" and self._compute_pending:
                        barriers.append(BarrierType.COMPUTE)
                        self._compute_pending = False

                result.append(ScheduledOp(
                    op="dma",
                    barriers_before=barriers,
                    dma_info=op,
                ))

                if kind == "load":
                    self._last_writer[buf_name] = "dma"
                    self._dma_pending = True

        return result

    def _needs_barrier(
        self,
        buf_name: str,
        current_idx: int,
        accesses: dict[str, list[tuple[int, bool, str]]],
    ) -> bool:
        """Check if reading *buf_name* at *current_idx* needs a DMA barrier."""
        hist = accesses.get(buf_name, [])
        for idx, is_writer, kind in hist:
            if idx >= current_idx:
                break
            if kind == "load" and is_writer:
                return True  # A previous load wrote this buffer
        return False
