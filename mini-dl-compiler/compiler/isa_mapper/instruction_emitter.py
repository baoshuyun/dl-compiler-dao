"""Instruction emitter: ScheduledOp → NPUInstruction sequence.

Translates each ScheduledOp in the linear schedule into one or more
concrete NPUInstruction objects with correct opcodes and fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .isa import (
    ACT_NONE,
    BAR_DMA,
    BAR_COMPUTE,
    OP_BARRIER,
    OP_COMPUTE,
    OP_LOAD,
    OP_LOOP,
    OP_NOP,
    OP_STORE,
    encode_barrier,
    encode_compute,
    encode_load,
    encode_loop,
    encode_nop,
    encode_store,
)
from .types import (
    ActType,
    BarrierType,
    BufferAssignment,
    HardwareConfig,
    NPUInstruction,
    ScheduledOp,
    TileOp,
)


@dataclass
class InstructionEmitter:
    """Emits NPU instructions from scheduled operations.

    Usage::

        emitter = InstructionEmitter(config, assignments)
        program = emitter.emit(scheduled_ops)
    """

    config: HardwareConfig = field(default_factory=HardwareConfig)
    bank_assignments: dict[str, BufferAssignment] = field(default_factory=dict)

    def emit(self, scheduled: list[ScheduledOp]) -> list[NPUInstruction]:
        """Convert a linear schedule into NPU instructions.

        Args:
            scheduled: Ordered list of ScheduledOp from the Scheduler.

        Returns:
            Ordered list of NPUInstruction objects.
        """
        instructions: list[NPUInstruction] = []

        for sched_op in scheduled:
            # 1. Emit barrier instructions
            for bar_type in sched_op.barriers_before:
                instructions.append(self._emit_barrier(bar_type))

            # 2. Emit the actual operation
            if isinstance(sched_op.op, TileOp):
                instructions.extend(self._emit_compute(sched_op.op))
            elif sched_op.op == "dma" and sched_op.dma_info:
                dma = sched_op.dma_info
                if dma["type"] == "load":
                    instructions.append(self._emit_load(dma))
                elif dma["type"] == "store":
                    instructions.append(self._emit_store(dma))

        # Finalize with a NOP sentinel
        instructions.append(NPUInstruction(
            opcode=OP_NOP,
            binary=encode_nop(),
            comment="halt",
        ))

        return instructions

    # ── Private emitters ───────────────────────────────────────

    def _emit_compute(self, tile: TileOp) -> list[NPUInstruction]:
        """Emit one or more COMPUTE instructions for a tile."""
        a_assign = self.bank_assignments.get(tile.a_name)
        b_assign = self.bank_assignments.get(tile.b_name)
        c_assign = self.bank_assignments.get(tile.c_name)

        a_src = a_assign.bank if a_assign else 0
        b_src = b_assign.bank if b_assign else 1
        c_dst = c_assign.bank if c_assign else 2

        act = ACT_NONE if tile.act == ActType.NONE else 0x1
        mat_dim = tile.mat_dim.value

        binary = encode_compute(a_src, b_src, c_dst, mat_dim, act)
        mi, mj, mk = tile.tile_idx

        comment = (
            f"COMPUTE {tile.mat_dim.name} "
            f"A[{tile.a_slice[0]},{tile.a_slice[1]}] @ "
            f"B[{tile.b_slice[0]},{tile.b_slice[1]}] → "
            f"C[{tile.c_slice[0]},{tile.c_slice[1]}] "
            f"tile({mi},{mj},{mk})"
        )
        if tile.act == ActType.RELU:
            comment += " + ReLU"

        return [NPUInstruction(opcode=OP_COMPUTE, binary=binary, comment=comment)]

    def _emit_load(self, dma: dict[str, Any]) -> NPUInstruction:
        bank = dma.get("bank", 0)
        ext_addr = dma.get("ext_addr", 0)
        size = dma.get("size", 0)
        binary = encode_load(bank, ext_addr, size)
        return NPUInstruction(
            opcode=OP_LOAD,
            binary=binary,
            comment=f"LOAD bank{bank}, size={size}, ext_addr={ext_addr}",
        )

    def _emit_store(self, dma: dict[str, Any]) -> NPUInstruction:
        bank = dma.get("bank", 0)
        ext_addr = dma.get("ext_addr", 0)
        size = dma.get("size", 0)
        binary = encode_store(bank, ext_addr, size)
        return NPUInstruction(
            opcode=OP_STORE,
            binary=binary,
            comment=f"STORE bank{bank}, size={size}, ext_addr={ext_addr}",
        )

    def _emit_barrier(self, bar_type: BarrierType) -> NPUInstruction:
        if bar_type == BarrierType.DMA:
            bar_val = BAR_DMA
            comment = "BARRIER DMA"
        elif bar_type == BarrierType.COMPUTE:
            bar_val = BAR_COMPUTE
            comment = "BARRIER COMPUTE"
        else:
            bar_val = 0x3
            comment = "BARRIER ALL"
        return NPUInstruction(
            opcode=OP_BARRIER,
            binary=encode_barrier(bar_val),
            comment=comment,
        )
