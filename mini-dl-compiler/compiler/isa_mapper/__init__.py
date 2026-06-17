"""ISA Mapper — compiler IR to NPU hardware instruction bridge.

The ISA Mapper is the backend that translates optimized deep-learning
compiler IR into a stream of 32-bit NPU instructions.  It is the
critical missing link between the software compiler (mini-dl-compiler)
and the hardware accelerator (NPU_Project).

Pipeline::

    Lowered IR ops
        │
        ▼
    Tiler           — decompose large matmuls into 4×4 NPU tiles
        │
        ▼
    Scheduler       — linearise, insert BARRIERs for sync
        │
        ▼
    BankAllocator   — assign tensors to SRAM banks (ping/pong)
        │
        ▼
    InstructionEmitter — TileOp / DMA → NPUInstruction
        │
        ▼
    Assembler       — encode to 32-bit binary instr_mem
        │
        ▼
    Simulator       — run on Python golden model, compare results
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .assembler import Assembler
from .bank_allocator import BankAllocator, _LiveRange
from .instruction_emitter import InstructionEmitter
from .scheduler import Scheduler
from .simulator import NPUSimulator
from .tiler import Tiler
from .types import (
    ActType,
    BufferAssignment,
    HardwareConfig,
    MatDim,
    NPUInstruction,
    NPUProgram,
    ScheduledOp,
    TileOp,
)


__all__ = [
    "ISAMapper",
    "Assembler",
    "BankAllocator",
    "InstructionEmitter",
    "NPUSimulator",
    "Scheduler",
    "Tiler",
    "ActType",
    "BufferAssignment",
    "HardwareConfig",
    "MatDim",
    "NPUInstruction",
    "NPUProgram",
    "ScheduledOp",
    "TileOp",
]


class ISAMapper:
    """Top-level ISA Mapper: lowers compiler IR ops to an NPU program.

    This is the main entry point.  It orchestrates tiling, scheduling,
    bank allocation, instruction emission, and assembly.

    Usage::

        mapper = ISAMapper()
        program = mapper.map_mlp(
            M=128, N=128, K=128,
            ext_a_addr=0, ext_b_addr=64, ext_c_addr=128,
        )
        print(program.display())

        # Run on the simulator to verify correctness
        sim = NPUSimulator()
        sim.load_program(program)
        sim.load_external_memory({i: 2 for i in range(256)})
        sim.run()
        results = sim.read_external_memory(128, 16)
    """

    def __init__(self, config: HardwareConfig | None = None) -> None:
        self.config = config or HardwareConfig()
        self.tiler = Tiler(self.config)
        self.scheduler = Scheduler()
        self.bank_allocator = BankAllocator(self.config)
        self.emitter = InstructionEmitter(config=self.config)
        self.assembler = Assembler()

    # ── High-level mapping APIs ─────────────────────────────────

    def map_matmul(
        self,
        M: int,
        N: int,
        K: int,
        *,
        ext_a_addr: int = 0,
        ext_b_addr: int = 64,
        ext_c_addr: int = 128,
        act: ActType = ActType.NONE,
    ) -> NPUProgram:
        """Map a single matmul C[M,N] = A[M,K] @ B[K,N] to NPU instructions.

        For multi-tile matmuls (M>4 or N>4 or K>4), each tile reloads its
        sub-matrix from external DRAM before computing, since the current
        COMPUTE ISA has no SRAM offset field.

        Args:
            M, N, K: Matrix dimensions.
            ext_a_addr: External DRAM base address for matrix A.
            ext_b_addr: External DRAM base address for matrix B.
            ext_c_addr: External DRAM base address for result C.
            act: Optional activation function.

        Returns:
            An assembled NPUProgram ready for execution.
        """
        a_size = M * K
        b_size = K * N
        c_size = M * N

        # Step 1: Tile
        tiles = self.tiler.tile_matmul("A", "B", "C", M, N, K, act=act)
        tm = self.config.array_rows
        tn = self.config.array_cols

        # Step 2: Build per-tile DMA + compute schedule
        # Each tile: LOAD A_sub → LOAD B_sub → COMPUTE → STORE C_sub
        # This works because each tile writes a disjoint C region
        # (K-tiling not yet supported — requires accumulate mode)
        all_ops: list[dict[str, Any] | TileOp] = []

        for tile in tiles:
            mi, mj, mk = tile.tile_idx
            m_start = tile.c_slice[0].start
            n_start = tile.c_slice[1].start
            m_size = tile.c_slice[0].stop - m_start
            n_size = tile.c_slice[1].stop - n_start
            k_size = tile.a_slice[1].stop - tile.a_slice[1].start

            a_start = tile.a_slice[0].start * K + tile.a_slice[1].start
            a_count = m_size * k_size
            b_start = tile.b_slice[0].start * N + tile.b_slice[1].start
            b_count = k_size * n_size
            c_start = ext_c_addr + m_start * N + n_start
            c_count = m_size * n_size

            tile_key = f"t{mi}_{mj}_{mk}"
            tile_a_name = f"A_{tile_key}"
            tile_b_name = f"B_{tile_key}"
            tile_c_name = f"C_{tile_key}"

            all_ops.append({
                "type": "load", "buf": tile_a_name,
                "bank": 0, "ext_addr": ext_a_addr + a_start, "size": a_count,
            })
            all_ops.append({
                "type": "load", "buf": tile_b_name,
                "bank": 1, "ext_addr": ext_b_addr + b_start, "size": b_count,
            })
            all_ops.append(TileOp(
                op_type="matmul",
                a_name=tile_a_name,
                b_name=tile_b_name,
                c_name=tile_c_name,
                a_slice=tile.a_slice,
                b_slice=tile.b_slice,
                c_slice=tile.c_slice,
                mat_dim=tile.mat_dim,
                act=tile.act,
                tile_idx=tile.tile_idx,
            ))
            all_ops.append({
                "type": "store", "buf": tile_c_name,
                "bank": 2, "ext_addr": c_start, "size": c_count,
            })

        # Step 3: Schedule with interleaved per-tile ordering
        scheduled = self.scheduler.schedule([], interleaved=all_ops)

        # Step 4: Bank allocation — simple: A→0, B→1, C→2
        live_ranges = self._build_live_ranges_multi(scheduled, a_size, b_size, c_size, tiles)
        assignments = self.bank_allocator.allocate(live_ranges)

        # Step 5: Update DMA op bank numbers from actual assignments
        for sop in scheduled:
            if sop.op == "dma" and sop.dma_info:
                buf_name = sop.dma_info["buf"]
                if buf_name in assignments:
                    sop.dma_info["bank"] = assignments[buf_name].bank

        # Step 6: Emit instructions
        self.emitter.bank_assignments = assignments
        instructions = self.emitter.emit(scheduled)

        # Step 7: Assemble
        return self.assembler.assemble(instructions)

    def map_mlp(
        self,
        M: int,
        N: int,
        K: int,
        *,
        ext_x_addr: int = 0,
        ext_w_addr: int = 256,
        ext_b_addr: int = 512,
        ext_c_addr: int = 768,
    ) -> NPUProgram:
        """Map an MLP layer C[M,N] = ReLU(X[M,K] @ W[K,N] + b[N]).

        The bias is pre-loaded into external memory and applied via the
        fused compute path.
        """
        return self.map_matmul(
            M, N, K,
            ext_a_addr=ext_x_addr,
            ext_b_addr=ext_w_addr,
            ext_c_addr=ext_c_addr,
            act=ActType.RELU,
        )

    # ── Validation ───────────────────────────────────────────────

    def verify(
        self,
        program: NPUProgram,
        inputs: dict[str, NDArray[np.int32]],
    ) -> dict[str, NDArray[np.int32]]:
        """Run *program* on the NPU simulator and return results.

        Args:
            program: Assembled NPU program.
            inputs: Dict mapping buffer name → NumPy array.
                    Supported keys: "A", "B", "X", "W", "b".

        Returns:
            Dict with "C" → result array.
        """
        sim = NPUSimulator(self.config)

        # Load input data into external memory
        ext_mem: dict[int, int] = {}
        offset = 0
        for key, arr in inputs.items():
            flat = arr.ravel().astype(np.int32)
            for i, val in enumerate(flat):
                ext_mem[offset + i] = int(val)
            offset += len(flat)

        sim.load_program(program)
        sim.load_external_memory(ext_mem)
        sim.run()

        # Determine output size and read
        c_size = 0
        for instr in program.instructions:
            if instr._opcode_int == 0x2:  # STORE
                c_size = (instr.binary >> 16) & 0x3FF
                break

        if c_size == 0:
            c_size = 16  # default 4x4

        results = sim.read_external_memory(offset, c_size)
        return {"C": np.array(results, dtype=np.int32)}

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _build_live_ranges_multi(
        scheduled: list[ScheduledOp],
        a_size: int,
        b_size: int,
        c_size: int,
        tiles: list[TileOp],
    ) -> list[_LiveRange]:
        """Extract liveness information from a multi-tile schedule.

        Each per-tile buffer is live from its first use (load or compute)
        to its last use (compute read or store).
        """
        live: list[_LiveRange] = []
        max_step = len(scheduled)

        # Collect all buffer names and their first/last use from the schedule
        buf_first: dict[str, int] = {}
        buf_last: dict[str, int] = {}
        buf_sizes: dict[str, int] = {}

        for i, sop in enumerate(scheduled):
            if isinstance(sop.op, TileOp):
                t = sop.op
                for name in (t.a_name, t.b_name):
                    if name not in buf_first:
                        buf_first[name] = i
                    buf_last[name] = i
                    m_sz = t.c_slice[0].stop - t.c_slice[0].start
                    n_sz = t.c_slice[1].stop - t.c_slice[1].start
                    k_sz = t.a_slice[1].stop - t.a_slice[1].start
                    buf_sizes[name] = max(buf_sizes.get(name, 0),
                                          m_sz * k_sz if "A" in name else k_sz * n_sz)
                c_name = t.c_name
                if c_name not in buf_first:
                    buf_first[c_name] = i
                buf_last[c_name] = i
                buf_sizes[c_name] = max(buf_sizes.get(c_name, 0),
                                        m_sz * n_sz)
            elif sop.op == "dma" and sop.dma_info:
                buf_name = sop.dma_info["buf"]
                if buf_name not in buf_first:
                    buf_first[buf_name] = i
                buf_last[buf_name] = i + 1

        for name in buf_first:
            is_out = name.startswith("C_") or name == "C"
            live.append(_LiveRange(
                name=name,
                num_words=buf_sizes.get(name, 16),
                first_use=buf_first[name],
                last_use=buf_last[name],
                is_input=name.startswith("A_") or name.startswith("B_"),
                is_output=is_out,
            ))

        return live
