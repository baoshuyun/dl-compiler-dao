"""NPU ISA encoding constants.

Must stay in sync with isa/isa_defines.vh in NPU_Project.
"""

# ── Opcodes (bits [31:28]) ──────────────────────────────────────
OP_NOP     = 0x0   # No operation, stall 1 cycle
OP_LOAD    = 0x1   # DMA load: ext_dram → SRAM
OP_STORE   = 0x2   # DMA store: SRAM → ext_dram
OP_COMPUTE = 0x3   # Systolic array matrix multiply
OP_BARRIER = 0x4   # Synchronization barrier
OP_CONFIG  = 0x5   # Configure hardware parameters
OP_LOOP    = 0x6   # Loop N iterations

OPCODE_NAMES = {
    OP_NOP: "NOP", OP_LOAD: "LOAD", OP_STORE: "STORE",
    OP_COMPUTE: "COMPUTE", OP_BARRIER: "BARRIER",
    OP_CONFIG: "CONFIG", OP_LOOP: "LOOP",
}

# ── COMPUTE fields ───────────────────────────────────────────────
ACT_NONE = 0x0  # No activation
ACT_RELU = 0x1  # ReLU activation

MAT_4x4 = 0x0
MAT_2x2 = 0x1
MAT_1x1 = 0x2

# ── BARRIER types ────────────────────────────────────────────────
BAR_DMA     = 0x1
BAR_COMPUTE = 0x2
BAR_ALL     = 0x3

# ── Hardware limits ──────────────────────────────────────────────
INSTR_DEPTH    = 256
SRAM_BANKS     = 4
BANK_DEPTH     = 1024
ARRAY_ROWS     = 4
ARRAY_COLS     = 4
DATA_WIDTH     = 32
PIPELINE_CYCLES = ARRAY_ROWS + ARRAY_COLS - 1  # = 7


def encode_compute(a_src: int, b_src: int, c_dst: int,
                   mat_dim: int = MAT_4x4, act: int = ACT_NONE) -> int:
    """Pack a COMPUTE instruction into a 32-bit word."""
    return (OP_COMPUTE << 28) | (a_src << 26) | (b_src << 24) | \
           (c_dst << 22) | (mat_dim << 20) | (act << 18)


def encode_load(bank: int, ext_addr: int, size: int) -> int:
    """Pack a LOAD instruction."""
    return (OP_LOAD << 28) | (bank << 26) | (size << 16) | (ext_addr & 0xFFFF)


def encode_store(bank: int, ext_addr: int, size: int) -> int:
    """Pack a STORE instruction."""
    return (OP_STORE << 28) | (bank << 26) | (size << 16) | (ext_addr & 0xFFFF)


def encode_barrier(bar_type: int) -> int:
    """Pack a BARRIER instruction."""
    return (OP_BARRIER << 28) | (bar_type << 26)


def encode_loop(count: int) -> int:
    """Pack a LOOP instruction."""
    return (OP_LOOP << 28) | (count & 0xFFFF)


def encode_nop() -> int:
    """Pack a NOP instruction."""
    return OP_NOP << 28
