"""
MLIR Lowering Pipeline — NPU Backend.

Full lowering chain:
  linalg.matmul → tiling (4x4) → affine copy → npu.load/store/compute
  → barrier insertion → instruction assembly → binary program

Conforms to MLIR pass infrastructure patterns:
each pass takes an MLIR module, transforms it, and returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .npu_dialect import (
    Opcode, ActType, MatDim, BarrierType,
    NpuOp, NpuNop, NpuLoad, NpuStore,
    NpuCompute, NpuBarrier, NpuConfig, NpuLoop,
    NpuProgram,
)


# ── Hardware configuration ──────────────────────────────────────────

@dataclass
class HardwareConfig:
    """NPU hardware parameters (sync with isa_defines.vh and npu_top.v)."""
    array_rows: int = 4
    array_cols: int = 4
    num_banks: int = 4
    bank_depth: int = 1024
    instr_depth: int = 256
    pipeline_latency: int = 7  # rows + cols - 1
    data_width: int = 32

    @property
    def tile_size_m(self) -> int:
        return self.array_rows

    @property
    def tile_size_n(self) -> int:
        return self.array_cols


# ── MLIR-style Operation definitions ─────────────────────────────────

@dataclass
class LinalgMatmul:
    """linalg.matmul operation: C[M,N] = A[M,K] @ B[K,N].

    Attributes:
        lhs: SSA name for A matrix.
        rhs: SSA name for B matrix.
        result: SSA name for C matrix.
        m, n, k: Matrix dimensions.
        act: Optional fused activation type.
    """
    lhs: str
    rhs: str
    result: str
    m: int
    n: int
    k: int
    act: Any = None  # ActType, set by frontend passes

    def __repr__(self):
        act_suffix = f" +{self.act.name}" if hasattr(self.act, 'name') and self.act else ""
        return (f"linalg.matmul {self.result}[{self.m},{self.n}] "
                f"= {self.lhs}[{self.m},{self.k}] @ {self.rhs}[{self.k},{self.n}]"
                f"{act_suffix}")


@dataclass
class AffineCopy:
    """affine.copy: extract a tile from a memref."""
    src: str       # source memref name
    dst: str       # destination tile buffer name
    offsets: tuple[int, int]  # (row_offset, col_offset)
    sizes: tuple[int, int]    # (rows, cols)

    def __repr__(self):
        return (f"affine.copy {self.dst}[{self.sizes[0]},{self.sizes[1]}] "
                f"= {self.src}[{self.offsets[0]}:, {self.offsets[1]}:]")


@dataclass
class _LowerResult:
    """Output of TileToNpuOpsPass.

    ops: generated NPU instructions.
    c_partials: dict[(m_off,n_off), [c_addr1, c_addr2, ...]]
                — addresses of K-tile partial sums that need accumulation.
    """
    ops: list
    c_partials: dict


@dataclass
class NPUModule:
    """MLIR module containing NPU operations.

    Represents the state after lowering from linalg through affine
    to the npu dialect.
    """
    name: str
    ops: list = field(default_factory=list)
    ssA_values: dict[str, int] = field(default_factory=dict)  # name -> address

    def add_op(self, op):
        self.ops.append(op)
        return self

    def __repr__(self):
        return f"npu.module @{self.name} ({len(self.ops)} ops)"


# ── Pass 1: LinalgToTiles ───────────────────────────────────────────

class LinalgToTilesPass:
    """Decompose linalg.matmul into 4x4 tile operations.

    Iterates over M, N, K dimensions with hardware tile sizes,
    producing TileOp structures for each sub-multiplication.
    For K-tiling, the activation is applied only on the final K tile
    (partial sum accumulation).
    """

    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or HardwareConfig()

    def run(self, matmul: LinalgMatmul, act: ActType = ActType.NONE) -> list[TileOp]:
        tiles = []
        tm, tn, tk = self.config.tile_size_m, self.config.tile_size_n, self.config.tile_size_m

        for m_off in range(0, matmul.m, tm):
            for n_off in range(0, matmul.n, tn):
                for k_off in range(0, matmul.k, tk):
                    is_last_k = (k_off + tk >= matmul.k)
                    tile_act = act if is_last_k else ActType.NONE
                    tiles.append(TileOp(
                        a_off=(m_off, k_off),
                        b_off=(k_off, n_off),
                        c_off=(m_off, n_off),
                        m=min(tm, matmul.m - m_off),
                        n=min(tn, matmul.n - n_off),
                        k=min(tk, matmul.k - k_off),
                        a_stride=matmul.k,
                        b_stride=matmul.n,
                        c_stride=matmul.n,
                        act=tile_act,
                    ))
        return tiles


@dataclass
class TileOp:
    """A single NPU tile: C_tile[m,n] = A_tile[m,k] @ B_tile[k,n].

    Carries the original matrix dimensions for computing 2D→1D addresses.
    """
    a_off: tuple[int, int]
    b_off: tuple[int, int]
    c_off: tuple[int, int]
    m: int
    n: int
    k: int
    a_stride: int = 0   # K dimension of full matrix A (bytes between rows)
    b_stride: int = 0   # N dimension of full matrix B
    c_stride: int = 0   # N dimension of full matrix C
    act: ActType = ActType.NONE

    def a_row_addr(self, row: int, base: int) -> int:
        """Address of A tile row `row` in external memory."""
        return base + (self.a_off[0] + row) * self.a_stride + self.a_off[1]

    def b_row_addr(self, row: int, base: int) -> int:
        """Address of B tile row `row` in external memory."""
        return base + (self.b_off[0] + row) * self.b_stride + self.b_off[1]

    def c_row_addr(self, row: int, base: int) -> int:
        """Address of C tile row `row` in external memory."""
        return base + (self.c_off[0] + row) * self.c_stride + self.c_off[1]

    @property
    def a_contiguous(self) -> bool:
        """True if A tile rows are contiguous (k == a_stride)."""
        return self.k == self.a_stride

    @property
    def b_contiguous(self) -> bool:
        """True if B tile rows are contiguous (n == b_stride)."""
        return self.n == self.b_stride

    @property
    def load_a_size(self) -> int:
        return self.m * self.k

    @property
    def load_b_size(self) -> int:
        return self.k * self.n

    @property
    def store_c_size(self) -> int:
        return self.m * self.n


# ── Pass 2: TileToNpuOps ────────────────────────────────────────────

class TileToNpuOpsPass:
    """Lower tile operations to NPU dialect: LOAD/COMPUTE/STORE sequences.

    Each tile becomes:
        npu.load bank=0, addr=a_addr, size=a_size
        npu.load bank=1, addr=b_addr, size=b_size
        npu.barrier DMA
        npu.compute a=0, b=1, c=2, dim=4x4, act=(RELU|NONE)
        npu.barrier COMPUTE
        npu.store bank=2, addr=c_addr, size=c_size
        npu.barrier DMA
    """

    def run(self, tiles: list[TileOp],
            a_base: int = 0, b_base: int = 64, c_base: int = 128) -> _LowerResult:
        """Lower tiles to NPU ops with correct 2D→1D address mapping.

        Returns (_LowerResult):
            ops: list of NpuOp
            c_addrs: dict[c_off_tuple] = list of external memory addresses
                     where partial K-tile results were stored.
                     Caller must accumulate across K-tiles in the same group.
        """
        ops = []
        c_partials: dict[tuple[int, int], list[int]] = {}
        c_next = c_base

        for i, tile in enumerate(tiles):
            # Fixed bank allocation: A=0, B=1, C=2
            # No double-buffering — correctness first, speed later
            a_bank, b_bank, c_bank = 0, 1, 2

            # Load A tile rows
            if tile.a_contiguous:
                addr = tile.a_row_addr(0, a_base)
                ops.append(NpuLoad(bank=a_bank, base_addr=addr, size=tile.load_a_size))
            else:
                for r in range(tile.m):
                    addr = tile.a_row_addr(r, a_base)
                    ops.append(NpuLoad(bank=a_bank, base_addr=addr, size=tile.k))

            # Load B tile rows
            if tile.b_contiguous:
                addr = tile.b_row_addr(0, b_base)
                ops.append(NpuLoad(bank=b_bank, base_addr=addr, size=tile.load_b_size))
            else:
                for r in range(tile.k):
                    addr = tile.b_row_addr(r, b_base)
                    ops.append(NpuLoad(bank=b_bank, base_addr=addr, size=tile.n))

            ops.append(NpuBarrier(BarrierType.DMA))

            # Each K-tile gets a unique C output address for partial sum accumulation
            tile_c_addr = c_next
            c_next += tile.store_c_size
            key = (tile.c_off[0], tile.c_off[1])
            c_partials.setdefault(key, []).append(tile_c_addr)

            ops.append(NpuCompute(
                a_src=a_bank, b_src=b_bank, c_dst=c_bank,
                dim=MatDim.MAT_4x4, act=tile.act,
            ))
            ops.append(NpuBarrier(BarrierType.COMPUTE))

            if tile.a_contiguous and tile.b_contiguous:
                ops.append(NpuStore(bank=c_bank, base_addr=tile_c_addr, size=tile.store_c_size))
            else:
                store_addr = tile_c_addr
                for r in range(tile.m):
                    ops.append(NpuStore(bank=c_bank, base_addr=store_addr, size=tile.n))
                    store_addr += tile.n

            ops.append(NpuBarrier(BarrierType.DMA))

        return _LowerResult(ops, c_partials)


# ── Pass 3: NpuOpsToProgram ─────────────────────────────────────────

class NpuOpsToProgramPass:
    """Assemble NPU operations into a 256-entry program.

    Inserts npu.nop padding and validates instruction count <= 256.
    """

    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or HardwareConfig()

    def run(self, ops: list[NpuOp]) -> NpuProgram:
        if len(ops) > self.config.instr_depth:
            raise ValueError(
                f"Instruction count {len(ops)} exceeds "
                f"instr_depth {self.config.instr_depth}"
            )
        return NpuProgram(instructions=list(ops))


# ── Full lowering pipeline ───────────────────────────────────────────

@dataclass
class NPULoweringPipeline:
    """End-to-end lowering: linalg.matmul → NpuProgram.

    Usage:
        pipeline = NPULoweringPipeline()
        program = pipeline.lower(LinalgMatmul(...))
    """

    config: HardwareConfig = field(default_factory=HardwareConfig)
    tiling_pass: LinalgToTilesPass = field(init=False)
    lowering_pass: TileToNpuOpsPass = field(init=False)
    assembly_pass: NpuOpsToProgramPass = field(init=False)

    def __post_init__(self):
        self.tiling_pass = LinalgToTilesPass(self.config)
        self.lowering_pass = TileToNpuOpsPass()
        self.assembly_pass = NpuOpsToProgramPass(self.config)

    def lower(self, matmul: LinalgMatmul,
              act: ActType = ActType.NONE):
        """Run the full lowering pipeline.

        Returns:
            (NpuProgram, c_partials): program ready for execution, plus
            dict mapping (m_off,n_off)→[c_addr] for K-tile accumulation.
        """
        tiles = self.tiling_pass.run(matmul, act)
        result = self.lowering_pass.run(tiles)
        program = self.assembly_pass.run(result.ops)
        return program, result.c_partials


# ── Quick test ───────────────────────────────────────────────────────

def _test():
    """Verify pipeline produces a valid program for a 4x4 matmul."""
    pipeline = NPULoweringPipeline()
    matmul = LinalgMatmul(
        lhs="%A", rhs="%B", result="%C",
        m=4, n=4, k=4,
    )
    program, c_partials = pipeline.lower(matmul, act=ActType.RELU)

    # Expected: 1 tile → 7 ops (2 loads + 1 barrier + compute + barrier + store + barrier)
    assert len(program.instructions) == 7, (
        f"Expected 7 ops, got {len(program.instructions)}"
    )
    assert program.instructions[0].encode() != 0  # first op is not NOP
    assert len(c_partials) == 1, "Single tile should produce 1 C-partial group"

    instr_mem = program.assemble()
    assert len(instr_mem) <= 256
    assert sum(1 for w in instr_mem if w != 0) == 7  # 7 non-NOP entries

    print("[PASS] NPU Lowering Pipeline:")
    for op in program.instructions:
        print(f"  {op}")
    print(f"  → assembled {len(instr_mem)} words ({7} non-NOP entries)")


if __name__ == "__main__":
    _test()
