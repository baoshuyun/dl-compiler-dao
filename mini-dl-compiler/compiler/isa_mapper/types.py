"""Types for the NPU ISA Mapper.

Defines the data model for NPU instructions, programs, buffer assignments,
and hardware configuration. All types are immutable dataclasses with
validation on construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ── Enums ───────────────────────────────────────────────────────

class ActType(IntEnum):
    NONE = 0
    RELU = 1


class MatDim(IntEnum):
    M4x4 = 0
    M2x2 = 1
    M1x1 = 2

    @property
    def rows(self) -> int:
        return {MatDim.M4x4: 4, MatDim.M2x2: 2, MatDim.M1x1: 1}[self]

    @property
    def cols(self) -> int:
        return self.rows


class BarrierType(IntEnum):
    DMA = 1
    COMPUTE = 2
    ALL = 3


class Opcode(IntEnum):
    NOP     = 0x0
    LOAD    = 0x1
    STORE   = 0x2
    COMPUTE = 0x3
    BARRIER = 0x4
    CONFIG  = 0x5
    LOOP    = 0x6


# ── Data classes ─────────────────────────────────────────────────

@dataclass(frozen=True)
class HardwareConfig:
    """NPU hardware parameters.

    Attributes:
        array_rows: Systolic array rows (default 4).
        array_cols: Systolic array columns (default 4).
        sram_banks: Number of SRAM banks (default 4).
        bank_depth: Depth of each bank in 32-bit words (default 1024).
        data_width: Data width in bits (default 32).
        instr_depth: Instruction memory depth (default 256).
    """
    array_rows: int = 4
    array_cols: int = 4
    sram_banks: int = 4
    bank_depth: int = 1024
    data_width: int = 32
    instr_depth: int = 256

    def __post_init__(self) -> None:
        if self.instr_depth > 256:
            raise ValueError(f"instr_depth must be ≤256, got {self.instr_depth}")

    @property
    def pipeline_cycles(self) -> int:
        """Latency of the systolic array pipeline."""
        return self.array_rows + self.array_cols - 1

    @property
    def total_sram_words(self) -> int:
        """Total SRAM capacity in 32-bit words."""
        return self.sram_banks * self.bank_depth


@dataclass(frozen=True)
class NPUInstruction:
    """A single decoded NPU instruction.

    The *binary* field holds the packed 32-bit word.  Convenience
    properties extract fields for debugging / display.

    *opcode* may be an Opcode enum or raw int; both are accepted.
    """
    opcode: Opcode | int
    binary: int
    comment: str = ""

    @property
    def _opcode_int(self) -> int:
        return int(self.opcode)

    @property
    def mnemonic(self) -> str:
        try:
            return Opcode(self._opcode_int).name
        except ValueError:
            return f"UNK({self._opcode_int})"

    def __str__(self) -> str:
        base = f"{self.mnemonic:<8} 0x{self.binary:08X}"
        if self.comment:
            base += f"  ; {self.comment}"
        return base


@dataclass(frozen=True)
class NPUProgram:
    """A complete NPU instruction program ready to load into instr_mem.

    Attributes:
        instructions: Decoded instruction list.
        instr_mem: 256-element list of 32-bit instruction words.
    """
    instructions: tuple[NPUInstruction, ...]
    instr_mem: tuple[int, ...]  # length == 256

    def __post_init__(self) -> None:
        if len(self.instr_mem) != 256:
            raise ValueError(
                f"instr_mem must have exactly 256 entries, got {len(self.instr_mem)}"
            )

    @property
    def program_length(self) -> int:
        """Number of non-NOP instructions (excluding trailing NOPs)."""
        for i in range(len(self.instr_mem) - 1, -1, -1):
            if (self.instr_mem[i] >> 28) != int(Opcode.NOP):
                return i + 1
        return 0

    def display(self) -> str:
        """Return a human-readable listing."""
        lines = []
        for i, instr in enumerate(self.instructions):
            marker = "→ " if i < self.program_length else "  "
            lines.append(f"{marker}[{i:3d}] {instr}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BufferAssignment:
    """Maps a logical tensor to a physical SRAM bank region.

    Attributes:
        name: Logical buffer name (e.g. "A_tile_0").
        bank: Physical bank index (0–3).
        base_addr: Starting word offset within the bank.
        num_words: Number of 32-bit words allocated.
    """
    name: str
    bank: int
    base_addr: int
    num_words: int

    def __post_init__(self) -> None:
        if self.bank not in (0, 1, 2, 3):
            raise ValueError(f"bank must be 0-3, got {self.bank}")
        if self.num_words <= 0:
            raise ValueError(f"num_words must be >0, got {self.num_words}")


@dataclass
class TileOp:
    """A single tiled compute operation.

    Represents one NPU COMPUTE instruction's worth of work: a 4×4 (or
    smaller) sub-matrix multiply-accumulate with optional activation.
    """
    op_type: str                                    # "matmul", "fused_mma_bias_relu"
    a_name: str                                     # Logical buffer name for matrix A
    b_name: str                                     # Logical buffer name for matrix B
    c_name: str                                     # Logical buffer name for result C
    a_slice: tuple[slice, slice]                    # (row_slice, col_slice) into A
    b_slice: tuple[slice, slice]                    # (row_slice, col_slice) into B
    c_slice: tuple[slice, slice]                    # destination slice in C
    mat_dim: MatDim = MatDim.M4x4
    act: ActType = ActType.NONE
    tile_idx: tuple[int, int, int] = field(default=(0, 0, 0))  # (m_tile, n_tile, k_tile)


@dataclass
class ScheduledOp:
    """An operation in the linear schedule with synchronization info."""
    op: TileOp | str                                  # TileOp for compute, str for DMA/BARRIER
    barriers_before: list[BarrierType] = field(default_factory=list)
    dma_info: dict[str, Any] | None = None            # For LOAD/STORE: {bank, ext_addr, size, is_load}
