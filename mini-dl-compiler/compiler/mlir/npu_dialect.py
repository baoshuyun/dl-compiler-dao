"""
NPU Dialect — MLIR-compatible Python implementation.

Seven hardware opcodes mapped to MLIR operations:
  npu.nop, npu.load, npu.store, npu.compute,
  npu.barrier, npu.config, npu.loop

Operand encoding matches isa_defines.vh bit fields.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


# === Opcode constants (sync with isa_defines.vh) ===

class Opcode(IntEnum):
    NOP     = 0x0
    LOAD    = 0x1
    STORE   = 0x2
    COMPUTE = 0x3
    BARRIER = 0x4
    CONFIG  = 0x5
    LOOP    = 0x6


# === Enum types ===

class ActType(IntEnum):
    """Activation function applied after compute."""
    NONE = 0
    RELU = 1


class MatDim(IntEnum):
    """Systolic array sub-array dimension."""
    MAT_4x4 = 0
    MAT_2x2 = 1
    MAT_1x1 = 2


class BarrierType(IntEnum):
    """Which engine to wait on before advancing PC."""
    DMA     = 0x1
    COMPUTE = 0x2
    ALL     = 0x3


# === NPU Dialect Operations ===

@dataclass
class NpuOp:
    """Base class for all NPU dialect operations."""
    opcode: Opcode

    def encode(self) -> int:
        """Pack this operation into a 32-bit instruction word."""
        raise NotImplementedError


@dataclass
class NpuNop(NpuOp):
    """No operation — advances PC by 1, no side effects."""

    def __init__(self):
        self.opcode = Opcode.NOP

    def encode(self) -> int:
        return self.opcode.value << 28

    def __repr__(self):
        return "npu.nop"


@dataclass
class NpuLoad(NpuOp):
    """DMA load: external DRAM -> SRAM bank.

    Transfer 'size' words from 'base_addr' in external memory
    into the specified SRAM 'bank' (0-3).
    """

    bank: int        # bits [27:26]
    base_addr: int   # bits [15:0]
    size: int        # bits [25:16]

    def __init__(self, bank: int, base_addr: int, size: int):
        assert 0 <= bank <= 3, f"bank {bank} out of range"
        assert 0 <= base_addr < 2**16, f"base_addr {base_addr} out of range"
        assert 0 < size < 2**10, f"size {size} out of range"
        self.opcode = Opcode.LOAD
        self.bank = bank
        self.base_addr = base_addr
        self.size = size

    def encode(self) -> int:
        return (
            (self.opcode.value << 28) |
            ((self.bank & 0x3) << 26) |
            ((self.size & 0x3FF) << 16) |
            (self.base_addr & 0xFFFF)
        )

    def __repr__(self):
        return f"npu.load bank={self.bank}, addr=0x{self.base_addr:04X}, size={self.size}"


@dataclass
class NpuStore(NpuOp):
    """DMA store: SRAM bank -> external DRAM."""

    bank: int
    base_addr: int
    size: int

    def __init__(self, bank: int, base_addr: int, size: int):
        assert 0 <= bank <= 3
        self.opcode = Opcode.STORE
        self.bank = bank
        self.base_addr = base_addr
        self.size = size

    def encode(self) -> int:
        return (
            (self.opcode.value << 28) |
            ((self.bank & 0x3) << 26) |
            ((self.size & 0x3FF) << 16) |
            (self.base_addr & 0xFFFF)
        )

    def __repr__(self):
        return f"npu.store bank={self.bank}, addr=0x{self.base_addr:04X}, size={self.size}"


@dataclass
class NpuCompute(NpuOp):
    """Systolic array matrix multiply with optional activation.

    A matrix: weights, flows left-to-right across rows.
    B matrix: activations, flows top-to-bottom down columns.
    C matrix: accumulated result, stays in place.
    Pipeline latency: ROWS + COLS - 1 (7 cycles for 4x4).
    """

    a_src: int        # source bank for A matrix
    b_src: int        # source bank for B matrix
    c_dst: int        # destination bank for C result
    dim: MatDim       # array dimension
    act: ActType      # post-compute activation

    def __init__(self, a_src: int, b_src: int, c_dst: int,
                 dim: MatDim = MatDim.MAT_4x4,
                 act: ActType = ActType.NONE):
        self.opcode = Opcode.COMPUTE
        self.a_src = a_src
        self.b_src = b_src
        self.c_dst = c_dst
        self.dim = dim
        self.act = act

    def encode(self) -> int:
        return (
            (self.opcode.value << 28) |
            ((self.a_src & 0x3) << 26) |
            ((self.b_src & 0x3) << 24) |
            ((self.c_dst & 0x3) << 22) |
            ((self.dim.value & 0x3) << 20) |
            ((self.act.value & 0x3) << 18)
        )

    def __repr__(self):
        return (f"npu.compute a_bank={self.a_src}, b_bank={self.b_src}, "
                f"c_bank={self.c_dst}, dim={self.dim.name}, act={self.act.name}")


@dataclass
class NpuBarrier(NpuOp):
    """Pipeline barrier: stall PC until target engine(s) idle.

    DMA_BARRIER: wait for DMA engine (LOAD/STORE done).
    COMPUTE_BARRIER: wait for systolic array (COMPUTE done).
    ALL_BARRIER: wait for both.
    """

    barrier_type: BarrierType

    def __init__(self, barrier_type: BarrierType):
        self.opcode = Opcode.BARRIER
        self.barrier_type = barrier_type

    def encode(self) -> int:
        return (
            (self.opcode.value << 28) |
            ((self.barrier_type.value & 0x3) << 26)
        )

    def __repr__(self):
        return f"npu.barrier {self.barrier_type.name}"


@dataclass
class NpuConfig(NpuOp):
    """Hardware configuration — writes parameter register (lower 16 bits)."""

    value: int

    def __init__(self, value: int):
        assert 0 <= value < 2**16
        self.opcode = Opcode.CONFIG
        self.value = value

    def encode(self) -> int:
        return (
            (self.opcode.value << 28) |
            (self.value & 0xFFFF)
        )

    def __repr__(self):
        return f"npu.config 0x{self.value:04X}"


@dataclass
class NpuLoop(NpuOp):
    """Hardware loop: repeat the immediately following instruction 'count' times.

    Nested loops are NOT supported — hardware only tracks one loop PC.
    """

    count: int

    def __init__(self, count: int):
        assert 0 < count < 2**16
        self.opcode = Opcode.LOOP
        self.count = count

    def encode(self) -> int:
        return (
            (self.opcode.value << 28) |
            (self.count & 0xFFFF)
        )

    def __repr__(self):
        return f"npu.loop x{self.count}"


# === NPU Program ===

@dataclass
class NpuProgram:
    """An assembled NPU program: 256 x 32-bit instruction memory."""
    instructions: list[NpuOp] = field(default_factory=list)

    @property
    def instr_mem(self) -> list[int]:
        """Compatibility: 256-entry 32-bit word array (NOP-padded).

        Matches the interface expected by NPUSimulator.load_program().
        """
        mem = [0 for _ in range(256)]
        for i, op in enumerate(self.instructions[:256]):
            mem[i] = op.encode()
        return mem

    def assemble(self) -> list[int]:
        """Pack instructions into a 256-entry 32-bit word array, NOP-padded."""
        return self.instr_mem

    def __repr__(self):
        return "\n".join(f"[{i:03d}] {op}" for i, op in enumerate(self.instructions))
