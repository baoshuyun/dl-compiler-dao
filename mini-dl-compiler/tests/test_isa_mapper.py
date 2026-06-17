"""Tests for the ISA Mapper module.

Validates: ISA encoding, tiling, scheduling, bank allocation,
instruction emission, assembly, and the end-to-end simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from compiler.isa_mapper.isa import (
    ACT_NONE,
    ACT_RELU,
    BAR_COMPUTE,
    BAR_DMA,
    MAT_4x4,
    MAT_2x2,
    MAT_1x1,
    OP_COMPUTE,
    OP_LOAD,
    OP_STORE,
    OP_BARRIER,
    OP_LOOP,
    OP_NOP,
    encode_barrier,
    encode_compute,
    encode_load,
    encode_loop,
    encode_nop,
    encode_store,
)
from compiler.isa_mapper.types import (
    ActType,
    HardwareConfig,
    MatDim,
    NPUProgram,
)
from compiler.isa_mapper.tiler import Tiler
from compiler.isa_mapper.scheduler import Scheduler
from compiler.isa_mapper.bank_allocator import BankAllocator, _LiveRange
from compiler.isa_mapper.instruction_emitter import InstructionEmitter
from compiler.isa_mapper.assembler import Assembler
from compiler.isa_mapper.simulator import NPUSimulator
from compiler.isa_mapper import ISAMapper


# ════════════════════════════════════════════════════════════════
# ISA encoding
# ════════════════════════════════════════════════════════════════

class TestISAEncoding:
    """Verify ISA instruction encoding matches the specification."""

    def test_nop_encoding(self) -> None:
        word = encode_nop()
        assert (word >> 28) == OP_NOP
        assert word == 0x00000000

    def test_load_encoding(self) -> None:
        word = encode_load(bank=1, ext_addr=0x0040, size=16)
        assert (word >> 28) == OP_LOAD
        assert ((word >> 26) & 0x3) == 1  # bank
        assert ((word >> 16) & 0x3FF) == 16  # size
        assert (word & 0xFFFF) == 0x0040  # ext_addr

    def test_store_encoding(self) -> None:
        word = encode_store(bank=2, ext_addr=0x0080, size=16)
        assert (word >> 28) == OP_STORE
        assert ((word >> 26) & 0x3) == 2

    def test_compute_encoding(self) -> None:
        word = encode_compute(a_src=0, b_src=1, c_dst=2, mat_dim=MAT_4x4, act=ACT_RELU)
        assert (word >> 28) == OP_COMPUTE
        assert ((word >> 26) & 0x3) == 0  # a_src
        assert ((word >> 24) & 0x3) == 1  # b_src
        assert ((word >> 22) & 0x3) == 2  # c_dst
        assert ((word >> 20) & 0x3) == MAT_4x4
        assert ((word >> 18) & 0x3) == ACT_RELU

    def test_barrier_encoding(self) -> None:
        word = encode_barrier(BAR_DMA)
        assert (word >> 28) == OP_BARRIER
        assert ((word >> 26) & 0x3) == BAR_DMA

    def test_loop_encoding(self) -> None:
        word = encode_loop(42)
        assert (word >> 28) == OP_LOOP
        assert (word & 0xFFFF) == 42


# ════════════════════════════════════════════════════════════════
# Tiler
# ════════════════════════════════════════════════════════════════

class TestTiler:
    """Verify tiling decomposes matrices correctly."""

    def test_small_matmul_one_tile(self) -> None:
        tiler = Tiler()
        tiles = tiler.tile_matmul("A", "B", "C", M=4, N=4, K=4)
        assert len(tiles) == 1
        t = tiles[0]
        assert t.mat_dim == MatDim.M4x4
        assert t.tile_idx == (0, 0, 0)

    def test_large_matmul_multiple_tiles(self) -> None:
        tiler = Tiler()
        tiles = tiler.tile_matmul("A", "B", "C", M=8, N=8, K=8)
        # 8/4 = 2 tiles per dimension → 2*2*2 = 8 tiles
        assert len(tiles) == 8
        # Verify tile indices cover full space
        indices = {t.tile_idx for t in tiles}
        assert len(indices) == 8
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    assert (i, j, k) in indices

    def test_rectangular_matmul(self) -> None:
        tiler = Tiler()
        tiles = tiler.tile_matmul("A", "B", "C", M=6, N=4, K=6)
        # M=6→2 tiles, N=4→1 tile, K=6→2 tiles = 4 tiles
        assert len(tiles) == 4

    def test_activation_on_last_k_tile(self) -> None:
        tiler = Tiler()
        tiles = tiler.tile_matmul("A", "B", "C", M=4, N=4, K=8, act=ActType.RELU)
        # K=8 → 2 k-tiles; only the last one has ReLU
        assert len(tiles) == 2
        assert tiles[0].act == ActType.NONE  # first K-tile, no activation
        assert tiles[1].act == ActType.RELU  # last K-tile, apply ReLU


# ════════════════════════════════════════════════════════════════
# Scheduler
# ════════════════════════════════════════════════════════════════

class TestScheduler:
    """Verify BARRIER insertion logic."""

    def test_simple_schedule_no_barriers(self) -> None:
        tiler = Tiler()
        tiles = tiler.tile_matmul("A", "B", "C", M=4, N=4, K=4)
        scheduler = Scheduler()
        # No DMA ops → no barriers needed for single tile
        scheduled = scheduler.schedule(tiles)
        assert len(scheduled) >= 1
        assert len(scheduled[0].barriers_before) == 0

    def test_schedule_with_dma_inserts_barriers(self) -> None:
        tiler = Tiler()
        tiles = tiler.tile_matmul("A", "B", "C", M=4, N=4, K=4)
        dma_ops = [
            {"type": "load", "buf": "A", "bank": 0, "ext_addr": 0, "size": 16},
            {"type": "load", "buf": "B", "bank": 1, "ext_addr": 64, "size": 16},
        ]
        scheduler = Scheduler()
        scheduled = scheduler.schedule(tiles, extra_ops=dma_ops)
        # The compute must wait for both loads
        compute_ops = [s for s in scheduled if isinstance(s.op, type(tiles[0]))]
        assert len(compute_ops) >= 1
        # At least one barrier before first compute to wait for DMA loads
        # (implementation may insert barriers at different points)
        has_barrier = any(
            len(s.barriers_before) > 0
            for s in scheduled
        )
        assert has_barrier, "Expected at least one BARRIER before compute after DMA loads"


# ════════════════════════════════════════════════════════════════
# Bank Allocator
# ════════════════════════════════════════════════════════════════

class TestBankAllocator:
    """Verify SRAM bank assignment."""

    def test_basic_allocation(self) -> None:
        alloc = BankAllocator()
        ranges = [
            _LiveRange("A", 16, 0, 10, is_input=True),
            _LiveRange("B", 16, 0, 10, is_input=True),
            _LiveRange("C", 16, 3, 10, is_output=True),
        ]
        assignments = alloc.allocate(ranges)
        assert "A" in assignments
        assert "B" in assignments
        assert "C" in assignments
        # A should go to bank 0 (input A → preferred bank 0)
        assert assignments["A"].bank == 0
        # B should go to bank 1 (input B → preferred bank 1)
        assert assignments["B"].bank == 1

    def test_output_to_scratchpad_banks(self) -> None:
        alloc = BankAllocator()
        ranges = [
            _LiveRange("C", 16, 0, 5, is_output=True),
        ]
        assignments = alloc.allocate(ranges)
        assert assignments["C"].bank in (2, 3)

    def test_rejects_overflow(self) -> None:
        alloc = BankAllocator(HardwareConfig(bank_depth=16))
        ranges = [
            _LiveRange("huge_A", 100, 0, 10),
        ]
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            alloc.allocate(ranges)


# ════════════════════════════════════════════════════════════════
# Assembler
# ════════════════════════════════════════════════════════════════

class TestAssembler:
    """Verify instruction assembly."""

    def test_assembles_to_256_words(self) -> None:
        from compiler.isa_mapper.types import NPUInstruction
        from compiler.isa_mapper.isa import OP_LOAD, OP_NOP, encode_load, encode_nop

        instrs = [
            NPUInstruction(opcode=OP_LOAD, binary=encode_load(0, 0, 16), comment="load A"),
            NPUInstruction(opcode=OP_NOP, binary=encode_nop(), comment="halt"),
        ]
        asm = Assembler()
        program = asm.assemble(instrs)
        assert len(program.instr_mem) == 256
        # program_length counts non-NOP instructions (last instr is NOP halt)
        assert program.program_length == 1
        # First word is LOAD, second is NOP, rest are padding NOPs
        assert (program.instr_mem[0] >> 28) == OP_LOAD
        assert (program.instr_mem[1] >> 28) == OP_NOP

    def test_rejects_too_many_instructions(self) -> None:
        from compiler.isa_mapper.types import NPUInstruction
        asm = Assembler()
        instrs = [NPUInstruction(opcode=OP_NOP, binary=0) for _ in range(300)]
        with pytest.raises(ValueError, match="Too many"):
            asm.assemble(instrs)


# ════════════════════════════════════════════════════════════════
# Simulator
# ════════════════════════════════════════════════════════════════

class TestSimulator:
    """Verify the NPU behavioral simulator."""

    def test_identity_matmul_simulator(self) -> None:
        """4×4 identity matrix: I @ I = I, then ReLU."""
        mapper = ISAMapper()
        program = mapper.map_matmul(M=4, N=4, K=4, act=ActType.RELU)

        # Load identity values: A[4×4]=1, B[4×4]=1
        ext_data = {}
        for i in range(16):
            ext_data[i] = 1       # A: addresses 0–15
            ext_data[64 + i] = 1  # B: addresses 64–79

        sim = NPUSimulator()
        sim.load_program(program)
        sim.load_external_memory(ext_data)
        sim.run()

        # C should be at addresses 128–143
        results = sim.read_external_memory(128, 16)
        # Each C[i] = sum(A[row] * B[col]) for 4×4 identity = 4
        # ReLU(max(0, 4)) = 4
        assert all(r == 4 for r in results), f"Expected all 4s, got {results}"

    def test_simulator_produces_done(self) -> None:
        mapper = ISAMapper()
        program = mapper.map_matmul(M=4, N=4, K=4)
        sim = NPUSimulator()
        sim.load_program(program)
        sim.load_external_memory({i: 1 for i in range(256)})
        sim.run()
        assert sim.done
        assert sim.total_cycles > 0

    def test_negative_relu_clamped(self) -> None:
        """ReLU should clamp negative results to 0."""
        mapper = ISAMapper()
        program = mapper.map_matmul(M=4, N=4, K=4, act=ActType.RELU)

        ext_data = {}
        # A = -1, B = 1 → result = -4 per element → ReLU → 0
        for i in range(16):
            ext_data[i] = -1
            ext_data[64 + i] = 1

        sim = NPUSimulator()
        sim.load_program(program)
        sim.load_external_memory(ext_data)
        sim.run()

        results = sim.read_external_memory(128, 16)
        assert all(r >= 0 for r in results), f"ReLU should clamp negatives: {results}"


# ════════════════════════════════════════════════════════════════
# End-to-End ISAMapper
# ════════════════════════════════════════════════════════════════

class TestISAMapperE2E:
    """End-to-end tests: compiler IR → NPU program → simulator → results."""

    def test_e2e_4x4_matmul(self) -> None:
        mapper = ISAMapper()
        program = mapper.map_matmul(M=4, N=4, K=4)

        assert program.program_length > 0
        assert program.program_length <= 256
        # Verify the program has the expected instruction types
        # (NOP is the sentinel; not counted by program_length)
        opcodes = {(w >> 28) for w in program.instr_mem[:program.program_length]}
        assert OP_LOAD in opcodes
        assert OP_COMPUTE in opcodes

    def test_e2e_verify_golden(self) -> None:
        """Verify simulated result against NumPy golden reference."""
        mapper = ISAMapper()
        M, N, K = 4, 4, 4

        A = np.ones((M, K), dtype=np.int32) * 2
        B = np.ones((K, N), dtype=np.int32) * 3

        program = mapper.map_matmul(M=M, N=N, K=K, act=ActType.RELU)

        sim = NPUSimulator()
        sim.load_program(program)

        ext_data: dict[int, int] = {}
        for i, v in enumerate(A.ravel()):
            ext_data[i] = int(v)
        for i, v in enumerate(B.ravel()):
            ext_data[64 + i] = int(v)

        sim.load_external_memory(ext_data)
        sim.run()

        # Read C from ext_addr 128
        results = np.array(sim.read_external_memory(128, M * N), dtype=np.int32)

        # Golden: C = ReLU(A @ B) = ReLU(all-2 @ all-3)
        # Each element = sum(2*3 for K=4) = 4*6 = 24
        golden = np.maximum(A.astype(np.int64) @ B.astype(np.int64), 0).astype(np.int32)
        np.testing.assert_array_equal(results.reshape(M, N), golden)

    def test_e2e_8x8_matmul(self) -> None:
        """Test a multi-tile matmul (M=8, N=4, K=4 — tiles only in M dimension).

        Note: K-tiling requires hardware accumulate mode (not yet in ISA).
        M-tiling and N-tiling work because each tile computes a disjoint
        output region.
        """
        mapper = ISAMapper()
        M, N, K = 8, 4, 4

        rng = np.random.RandomState(42)
        A = rng.randint(0, 5, (M, K)).astype(np.int32)
        B = rng.randint(0, 5, (K, N)).astype(np.int32)

        program = mapper.map_matmul(M=M, N=N, K=K)

        sim = NPUSimulator()
        sim.load_program(program)

        ext_data: dict[int, int] = {}
        for i, v in enumerate(A.ravel()):
            ext_data[i] = int(v)
        for i, v in enumerate(B.ravel()):
            ext_data[64 + i] = int(v)

        sim.load_external_memory(ext_data)
        sim.run()

        results = np.array(sim.read_external_memory(128, M * N), dtype=np.int32)
        golden = (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32)
        np.testing.assert_array_equal(results.reshape(M, N), golden)
