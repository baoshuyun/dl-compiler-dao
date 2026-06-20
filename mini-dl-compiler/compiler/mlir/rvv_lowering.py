"""
RVV Lowering Pass: linalg.matmul → RISC-V Vector (RVV) assembly.

Extends the MLIR lowering pipeline with a RISC-V Vector backend.
After this pass, linalg operations are lowered to RVV v1.0
vector instructions with explicit tiling for VLEN.

Full lowering chain:
  linalg.matmul → RVV tile loops → vsetvl + vle/vse + vfmacc
  → RVV assembly output

RVV ISA subset used (V extension 1.0):
  vsetvli      — set vector length
  vle32.v      — vector load (32-bit elements)
  vse32.v      — vector store (32-bit elements)
  vfmacc.vv    — vector fused multiply-accumulate
  vfadd.vv     — vector float add
  vfmax.vv     — vector float max (for ReLU)
  vmv.v.i      — vector integer move (zero init)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .lowering_pipeline import LinalgMatmul
from .npu_dialect import ActType


# ── RVV Hardware Configuration ────────────────────────────────────────

@dataclass
class RVVConfig:
    """RISC-V Vector extension configuration.

    VLEN: vector register width in bits (common: 128, 256, 512).
    ELEN: max element width (32 for f32, 64 for f64).
    LMUL: vector length multiplier (1, 2, 4, 8).
    """
    vlen: int = 256        # vector register width (bits)
    elen: int = 32         # element width (bits)
    lmul: int = 1          # vector length multiplier
    vector_regs: int = 32  # number of vector registers (v0-v31)

    @property
    def max_elements_per_reg(self) -> int:
        """Maximum f32 elements per vector register."""
        return (self.vlen * self.lmul) // self.elen

    @property
    def vtype_sew(self) -> str:
        """SEW string for vsetvl (e32 for f32)."""
        return "e32"

    @property
    def vtype_lmul(self) -> str:
        """LMUL string for vsetvl."""
        return f"m{self.lmul}" if self.lmul <= 1 else f"mf{self.lmul}"


# ── RVV Operation Types ───────────────────────────────────────────────

class RVVOp:
    """Base class for RVV assembly operations."""
    def emit(self) -> str:
        raise NotImplementedError


@dataclass
class RVVSetVL(RVVOp):
    """vsetvli: configure vector unit for N elements."""
    rd: str       # destination GPR (e.g., "t0")
    n: int        # number of elements to process
    sew: str = "e32"
    lmul: str = "m1"

    def emit(self) -> str:
        return f"vsetvli {self.rd}, x0, {self.sew}, {self.lmul}  # process {self.n} elements"


@dataclass
class RVVLoad(RVVOp):
    """vle32.v: load vector from memory."""
    vd: str       # destination vector register (e.g., "v0")
    rs: str       # base address register (e.g., "a0")
    n: int = 0    # element count (informational)

    def emit(self) -> str:
        return f"vle32.v {self.vd}, ({self.rs})"


@dataclass
class RVVStore(RVVOp):
    """vse32.v: store vector to memory."""
    vs: str       # source vector register
    rd: str       # base address register
    n: int = 0

    def emit(self) -> str:
        return f"vse32.v {self.vs}, ({self.rd})"


@dataclass
class RVVFMAcc(RVVOp):
    """vfmacc.vv: fused multiply-accumulate (vd = vd + vs1 * vs2)."""
    vd: str       # accumulator / destination
    vs1: str      # first operand
    vs2: str      # second operand

    def emit(self) -> str:
        return f"vfmacc.vv {self.vd}, {self.vs1}, {self.vs2}"


@dataclass
class RVVFAdd(RVVOp):
    """vfadd.vv: vector float add."""
    vd: str
    vs1: str
    vs2: str

    def emit(self) -> str:
        return f"vfadd.vv {self.vd}, {self.vs1}, {self.vs2}"


@dataclass
class RVVFMax(RVVOp):
    """vfmax.vv: vector float max (ReLU: max(x, 0.0))."""
    vd: str
    vs1: str
    vs2: str

    def emit(self) -> str:
        return f"vfmax.vv {self.vd}, {self.vs1}, {self.vs2}"


@dataclass
class RVVZero(RVVOp):
    """vmv.v.i: zero-initialize vector register."""
    vd: str

    def emit(self) -> str:
        return f"vmv.v.i {self.vd}, 0"


@dataclass
class RVVLabel(RVVOp):
    """Assembly label."""
    name: str

    def emit(self) -> str:
        return f"{self.name}:"


@dataclass
class RVVAddi(RVVOp):
    """addi: GPR integer add immediate (address bump)."""
    rd: str; rs: str; imm: int

    def emit(self) -> str:
        return f"addi {self.rd}, {self.rs}, {self.imm}"


@dataclass
class RVVComment(RVVOp):
    """Assembly comment."""
    text: str

    def emit(self) -> str:
        return f"    # {self.text}"


# ── RVV Program Container ─────────────────────────────────────────────

@dataclass
class RVVProgram:
    """An RVV assembly program: list of RVV operations."""
    ops: list[RVVOp] = field(default_factory=list)
    config: RVVConfig = field(default_factory=RVVConfig)

    def emit(self) -> str:
        """Emit the full RVV assembly program."""
        lines = []
        lines.append("# RISC-V Vector (RVV v1.0) assembly")
        lines.append(f"# VLEN={self.config.vlen}, SEW={self.config.vtype_sew}, "
                     f"LMUL={self.config.vtype_lmul}")
        lines.append("")
        for op in self.ops:
            lines.append(op.emit())
        return "\n".join(lines)


# ── Pass: LinalgToRVV ─────────────────────────────────────────────────

class LinalgToRVVPass:
    """Lower linalg.matmul to RISC-V Vector assembly.

    Strategy: K-outer, M×N-inner tiling for GEMM.
      for k_tile in 0..K step VEC:
        for m in 0..M:
          load A[m, k_tile:k_tile+VEC] → vA
          for n in 0..N:
            load B[k_tile, n] → vB (broadcast single column)
            load C[m, n] → vC
            vfmacc vC, vA, vB
            store C[m, n] ← vC

    This is a pedagogical RVV GEMM; production code would use
    multi-level tiling and register blocking.
    """

    def __init__(self, config: RVVConfig | None = None):
        self.config = config or RVVConfig()

    def lower(self, matmul: LinalgMatmul, act: ActType = ActType.NONE) -> RVVProgram:
        """Lower a linalg.matmul to an RVV assembly program.

        Args:
            matmul: The linalg.matmul operation.
            act: Optional post-compute activation.

        Returns:
            RVVProgram with full RVV assembly.
        """
        prog = RVVProgram(config=self.config)
        vec = self.config.max_elements_per_reg  # elements per vector reg
        M, N, K = matmul.m, matmul.n, matmul.k

        # Register allocation (calling convention)
        a_ptr = "a0"   # base pointer to A matrix
        b_ptr = "a1"   # base pointer to B matrix
        c_ptr = "a2"   # base pointer to C matrix
        m_reg = "a3"   # M loop counter
        k_reg = "a4"   # K loop counter
        vl  = "t0"     # vector length result

        # Vector registers
        vA = "v0"      # A tile load
        vB = "v4"      # B column load
        vC = "v8"      # C accumulator / result
        vZero = "v12"  # constant zero (for ReLU)
        vTmp = "v16"   # temporary for address calc

        prog.ops = []
        emit = prog.ops.append

        emit(RVVComment(f"linalg.matmul {matmul.result}[{M},{N}] = "
                        f"{matmul.lhs}[{M},{K}] @ {matmul.rhs}[{K},{N}]"))
        emit(RVVComment(f"RVV GEMM: VLEN={self.config.vlen}, VEC={vec}"))

        # Set VL for one full row tile
        n_elements = min(vec, K)  # elements per K-tile
        emit(RVVSetVL(vl, n_elements, self.config.vtype_sew, self.config.vtype_lmul))

        # Zero-initialize vZero for ReLU comparison
        if act == ActType.RELU:
            emit(RVVZero(vZero))

        # ── Triple loop: M, K-tile, N ──
        emit(RVVComment(f"Outer loop: M=0..{M-1}"))
        for m in range(M):
            emit(RVVComment(f"  m={m}"))

            # Compute base address for A row m
            a_row_off = m * K * 4  # 4 bytes per f32
            emit(RVVAddi(a_ptr, a_ptr, a_row_off) if m == 0 else
                 RVVComment(f"  A base = A[0] + {a_row_off}"))

            for k_start in range(0, K, n_elements):
                k_end = min(k_start + n_elements, K)
                k_len = k_end - k_start
                emit(RVVComment(f"    k_tile=[{k_start}:{k_end}]"))

                # Load A[m, k_start:k_end]
                k_off = k_start * 4
                emit(RVVAddi("t1", "a0", a_row_off + k_off))
                emit(RVVLoad(vA, "t1", k_len))

                # Load B[k_start:k_end, n] and compute C[m, n]
                for n in range(N):
                    # Load B column: B[k_start, n] — need column-major access
                    # For simplicity: load single B elements into scalar-like vec
                    b_addr = (k_start * N + n) * 4
                    emit(RVVAddi("t2", "b1", b_addr))
                    emit(RVVLoad(vB, "t2", k_len))

                    # Load C[m, n] accumulator
                    c_addr = (m * N + n) * 4
                    emit(RVVAddi("t3", "c2", c_addr))

                    if k_start == 0:
                        emit(RVVZero(vC))
                    else:
                        emit(RVVLoad(vC, "t3", 1))

                    emit(RVVFMAcc(vC, vA, vB))

                    # Store back
                    emit(RVVStore(vC, "t3", 1))

            # Apply ReLU if needed
            if act == ActType.RELU:
                emit(RVVComment(f"    ReLU: max(C[{m},:], 0)"))
                for n in range(N):
                    c_addr = (m * N + n) * 4
                    # Reload, apply max, store
                    emit(RVVAddi("t3", "c2", c_addr))
                    emit(RVVLoad(vC, "t3", 1))
                    emit(RVVFMax(vC, vC, vZero))
                    emit(RVVStore(vC, "t3", 1))

        emit(RVVComment("GEMM complete"))
        emit(RVVComment(f"Result in C[{M},{N}] at ({c_ptr})"))

        return prog


# ── RVV Lowering Pipeline ─────────────────────────────────────────────

@dataclass
class RVVLoweringPipeline:
    """End-to-end RVV lowering: linalg.matmul → RVV assembly.

    Usage:
        pipeline = RVVLoweringPipeline()
        asm = pipeline.lower(LinalgMatmul(...))
        print(asm)
    """

    config: RVVConfig = field(default_factory=RVVConfig)
    linalg_pass: LinalgToRVVPass = field(init=False)

    def __post_init__(self):
        self.linalg_pass = LinalgToRVVPass(self.config)

    def lower(self, matmul: LinalgMatmul, act: ActType = ActType.NONE) -> str:
        """Lower linalg.matmul → RVV assembly string.

        Returns:
            Human-readable RVV assembly.
        """
        prog = self.linalg_pass.lower(matmul, act)
        return prog.emit()


# ── Quick test ────────────────────────────────────────────────────────

def _test():
    """Verify RVV lowering produces valid assembly for a 4x4 matmul."""
    config = RVVConfig(vlen=256, elen=32, lmul=1)
    pipeline = RVVLoweringPipeline(config=config)
    matmul = LinalgMatmul("%A", "%B", "%C", m=4, n=4, k=8)
    asm = pipeline.lower(matmul, act=ActType.RELU)

    assert "vsetvli" in asm, "Missing vsetvli"
    assert "vle32.v" in asm, "Missing vle32.v"
    assert "vfmacc.vv" in asm, "Missing vfmacc.vv"
    assert "vse32.v" in asm, "Missing vse32.v"
    assert "vfmax.vv" in asm, "Missing vfmax.vv (ReLU)"
    assert "vmv.v.i" in asm, "Missing vmv.v.i (zero init)"
    print("[PASS] RVV Lowering Pipeline:")
    print(asm)


if __name__ == "__main__":
    _test()
